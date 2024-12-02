import os
from openai import OpenAI
import json
import logging
from copy import deepcopy
import time
import pandas as pd


class OpenAIBatchRunner:
    def __init__(
        self,
        openai_api_key: str,
        system_prompt: str,
        json_schema: dict | None = None,
        input_file: str = None,
        batch_input_folder: str = "../data/batch_temp/batch_input/",
        batch_output_folder: str = "../data/batch_temp/batch_output/",
        id_folder: str = "../data/batch_temp/ids/",
    ):
        self.client = OpenAI(api_key=openai_api_key)
        self.input_file = input_file

        # Directories for managing input/output and IDs
        self.batch_input_folder = batch_input_folder
        if not os.path.exists(batch_input_folder):
            os.makedirs(batch_input_folder)

        self.batch_output_folder = batch_output_folder
        if not os.path.exists(batch_output_folder):
            os.makedirs(batch_output_folder)

        self.id_folder = id_folder
        if not os.path.exists(id_folder):
            os.makedirs(id_folder)

        self.log = logging.getLogger(__name__)

        self.schema = None
        if json_schema:
            self.log.info("Utilizing structured output JSON Schema")
            self.schema = json_schema
            self.base_json = {
                "custom_id": None,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": None,
                    "max_tokens": 1024,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": json_schema,
                    },
                },
            }
        else:
            self.base_json = {
                "custom_id": None,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": "gpt-4o-mini", "messages": None, "max_tokens": 1024},
            }

        self.system_prompt = {
            "role": "system",
            "content": system_prompt,
        }

    def create_jsonl_batches(
        self,
        processing_function: None,
        processed_file_location: str = None,
        batch_size: int = 500,
    ):
        """
        Create JSONL batch files. Expects either an input file with columns "id" and "user_input", or
        an arbitrary data file with the appropriate processing function to produce a dataframe with "id"
        and "user_input" columns.

        Args:
            batch_size (int): Number of responses in each JSONL file.
        """
        self.log.info("Creating JSONL batch files")
        df = pd.read_csv(self.input_file, delimiter="\t", index_col=False)

        if processing_function:
            self.log.info("Processing input data using processing function...")
            df = processing_function(df, processed_file_location)

        if df.shape[1] != 2:
            self.log.error("Input dataframe must have 2 columns.")
            exit(1)

        num_batches = df.shape[0] // batch_size + (
            1 if df.shape[0] % batch_size != 0 else 0
        )

        # JSON reference
        batch_json = deepcopy(self.base_json)
        batch_json["body"]["messages"] = [self.system_prompt, None]

        for batch_num in range(num_batches):
            batch_list = []
            start_idx, end_idx = batch_num * batch_size, (batch_num + 1) * batch_size

            df_batch = df[start_idx:end_idx]
            self.log.info(
                f"Creating batch {batch_num} from {start_idx} to {start_idx + df_batch.shape[0] - 1}, inclusive"
            )

            for _, item in df_batch.iterrows():
                batch_json["custom_id"] = item["id"]
                batch_json["body"]["messages"][1] = {
                    "role": "user",
                    "content": [{"type": "text", "text": item["user_input"]}],
                }
                batch_list.append(deepcopy(batch_json))

            self.log.info(f"    Created {len(batch_list)} requests")
            self.log.info(f"    Writing batches to {self.batch_input_folder}")
            with open(
                f"{self.batch_input_folder}batch_{batch_num}.jsonl", mode="w"
            ) as caption_file:
                for request in batch_list:
                    caption_file.write(json.dumps(request) + "\n")

        self.log.info(f"Finished writing all {num_batches} files")

    def upload_batch_files(self):
        """
        Iterate through `self.batch_input_folder` and upload the files to OpenAI.
        Records OpenAI file IDs in `self.id_folder/fileids.txt`.
        """
        self.log.info("Uploading batch files...")
        openai_file_ids = []
        for batch_file_name in os.listdir(self.batch_input_folder):
            openai_file = self.client.files.create(
                file=open(f"{self.batch_input_folder}/{batch_file_name}", mode="rb"),
                purpose="batch",
            )
            openai_file_ids.append((openai_file.id, openai_file.filename))
            self.log.info(f"Uploaded {openai_file.filename} with id: {openai_file.id}")
            time.sleep(1)

        self.log.info(f"Finished uploading {len(openai_file_ids)} files")
        with open(f"{self.id_folder}/fileids.txt", mode="w") as f:
            for fileid, filename in openai_file_ids:
                f.write(f"{fileid}\t{filename}\n")

        self.log.info("Finished writing OpenAI file IDs locally")

    def submit_batch_jobs(self):
        """
        Submits batch jobs based on file IDs recorded in `self.id_folder`/fileids.txt.
        Records batch IDs locally in `self.id_folder/batchids.txt`
        """
        self.log.info("Submitting batch jobs...")
        file_ids = []
        with open(f"{self.id_folder}/fileids.txt", mode="r") as f:
            for data in f:
                file_entry = data.split("\t")
                file_ids.append((file_entry[0], file_entry[1].strip()))

        self.log.info(f"Retrieved {len(file_ids)} file ids from local file")
        batch_ids = []
        for file_id, file_name in file_ids:
            batch_job = self.client.batches.create(
                input_file_id=file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            batch_ids.append((batch_job.id, file_name))
            self.log.info(f"Submitted job for file {file_name} with ID: {batch_job.id}")
            time.sleep(1)

        self.log.info(f"Finished submitting {len(batch_ids)} jobs")

        with open(f"{self.id_folder}/batchids.txt", mode="w") as f:
            for batch_id, file_name in batch_ids:
                f.write(f"{batch_id}\t{file_name}\n")
        self.log.info("Finished writing OpenAI file IDs locally")

    def check_status_and_download(self):
        """
        Read in the batch IDs from local file.
        Periodically checks for the batch job status.
        Downloads the resulting JSONL when complete.
        """
        self.log.info("Checking status for batch jobs...")
        batch_ids = []
        with open(f"{self.id_folder}/batchids.txt", mode="r") as f:
            for data in f:
                file_entry = data.split("\t")
                batch_ids.append((file_entry[0], file_entry[1].strip()))

        self.log.info(f"Retrieved {len(batch_ids)} batch IDs from local storage")

        # clear out the output_fileids.txt file
        with open(f"{self.id_folder}/output_fileids.txt", mode="w") as f:
            f.write("")

        FAILED_STATUS = ["failed", "expired", "cancelled"]
        failed_batches = []
        while len(batch_ids) > 0:
            batch_indices_to_remove = []
            # go through the status of each batch job
            for i, (batch_id, file_name) in enumerate(batch_ids):
                job = self.client.batches.retrieve(batch_id)
                if job.status in FAILED_STATUS:
                    self.log.warning(
                        f'Batch with file {file_name} and ID "{batch_id}" has failed with status {job.status}'
                    )
                    failed_batches.append((batch_id, file_name))
                elif job.status == "completed":
                    self.log.info(
                        f"Batch for file {file_name} completed! Downloading data..."
                    )
                    result_file = self.client.files.content(job.output_file_id).content

                    with open(f"{self.id_folder}/output_fileids.txt", mode="a") as f:
                        f.write(f"{job.output_file_id}\t{file_name}\n")

                    with open(
                        f"{self.batch_output_folder}/output_{file_name}", mode="wb"
                    ) as f:
                        f.write(result_file)

                    batch_indices_to_remove.append(i)

                time.sleep(1)

            # remove batches from check list
            if len(batch_indices_to_remove) > 0:
                batch_ids = [
                    batch_id
                    for i, batch_id in enumerate(batch_ids)
                    if i not in batch_indices_to_remove
                ]

            if len(batch_ids) > 0:
                self.log.info("Sleeping for 5 minutes...")
                time.sleep(60.0 * 5.0)

        self.log.info("Finished retrieving data!")

    def delete_data_files(self):
        """
        Using the file_ids and batch_ids stored locally, delete them from OpenAI's file storage.
        """
        self.log.warning(
            "Starting deletion of input and output files stored in OpenAI's file storage..."
        )
        time.sleep(15)  # just in case you want to cancel

        file_ids = []
        with open(f"{self.id_folder}/fileids.txt", mode="r") as f:
            for data in f:
                file_entry = data.split("\t")
                file_ids.append((file_entry[0], file_entry[1].strip()))
        with open(f"{self.id_folder}/output_fileids.txt", mode="r") as f:
            for data in f:
                file_entry = data.split("\t")
                file_ids.append((file_entry[0], "output_" + file_entry[1].strip()))

        self.log.info(f"Retrieved {len(file_ids)} file IDs")

        for file_id, name in file_ids:
            self.log.info(f"Deleting {name} with ID {file_id}")
            self.client.files.delete(file_id)
            time.sleep(2)

        self.log.info("Finished deleting files in OpenAI storage")

    def get_data(self, file_prefix: str = "batch") -> dict:
        """
        Retrieve JSONL data and returns dictionary of data in the form {image_id: data}
        Args:
            batch_folder (str): Folder that stores output JSONL files
            file_prefix (str): String that is before each `_{batch_num}.jsonl` file
        """

        all_file_names = [
            f for f in sorted(os.listdir(self.batch_output_folder)) if file_prefix in f
        ]
        result_data = {}
        for file_name in all_file_names:
            file_path = os.path.join(self.batch_output_folder, file_name)
            with open(file_path, mode="r") as f:
                for line in f:
                    json_data = json.loads(line.rstrip())
                    obj_id = json_data["custom_id"]
                    response = json_data["response"]
                    if response["status_code"] != 200:
                        print(f"Warning! {obj_id} did not return response code 200")

                    choice = response["body"]["choices"][0]
                    output_content = choice["message"]["content"]
                    result_data[obj_id] = output_content

        return result_data
