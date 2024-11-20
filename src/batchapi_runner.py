import os
from openai import OpenAI
import json
import logging
from copy import deepcopy
import time
import pandas as pd
from tqdm import tqdm


class OpenAIBatchRunner:
    def __init__(
        self,
        openai_api_key: str,
        system_prompt: str,
        input_file: str,
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

    def create_jsonl_batches(self, batch_size: int = 500):
        """
        Create JSONL batch files.

        Args:
            batch_size (int): Number of responses in each JSONL file.
        """
        self.log.info("Creating JSONL batch files")
        df = pd.read_csv(self.input_file, delimiter="\t")
        df = df.sample(frac=1, random_state=42).reset_index(drop=True) # makes batches ~the same size

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

            for _, item in tqdm(df_batch.iterrows(), total=len(df_batch)):
                batch_json["custom_id"] = f"asin_{item["ASIN"]}"
                batch_json["body"]["messages"][1] = {
                    "role": "user",
                    "content": [{"type": "text", "text": item["Reviews"]}],
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
                file=open(f"{self.batch_input_folder}{batch_file_name}", mode="rb"),
                purpose="batch",
            )
            openai_file_ids.append((openai_file.id, openai_file.filename))
            self.log.info(f"Uploaded {openai_file.filename} with id: {openai_file.id}")
            time.sleep(1)

        self.log.info(f"Finished uploading {len(openai_file_ids)} files")
        with open(f"{self.id_folder}fileids.txt", mode="w") as f:
            for fileid, filename in openai_file_ids:
                f.write(f"{fileid}\t{filename}\n")

        self.log.info("Finished writing OpenAI file IDs locally")


    # def submit_batch_jobs(self):
    #     """
    #     Submits a single batch job at a time based on file IDs recorded in `self.id_folder`/fileids.txt.
    #     Waits for the status to complete before moving to the next job.
    #     """
    #     self.log.info("Submitting batch jobs one at a time...")
    #     file_ids = []
    #     with open(f"{self.id_folder}fileids.txt", mode="r") as f:
    #         for data in f:
    #             file_entry = data.split("\t")
    #             file_ids.append((file_entry[0], file_entry[1].strip()))

    #     self.log.info(f"Retrieved {len(file_ids)} file ids from local file")

    #     FAILED_STATUS = ["failed", "expired", "cancelled"]
    #     for file_id, file_name in file_ids:
    #         # Submit the batch job
    #         batch_job = self.client.batches.create(
    #             input_file_id=file_id,
    #             endpoint="/v1/chat/completions",
    #             completion_window="24h",
    #         )
    #         batch_id = batch_job.id
    #         self.log.info(f"Submitted job for file {file_name} with ID: {batch_id}")

    #         # Wait for the job to complete or fail
    #         while True:
    #             job = self.client.batches.retrieve(batch_id)
    #             if job.status in FAILED_STATUS:
    #                 self.log.warning(
    #                     f'    Batch with file {file_name} and ID "{batch_id}" has failed with status {job.status}'
    #                 )
    #                 self.log.info("    Now sleeping for 10 minutes to ensure that OpenAI's enqueued token count is reset correctly...")
    #                 time.sleep(60*5)
    #                 break
    #             elif job.status == "completed":
    #                 self.log.info(
    #                     f"    Batch for file {file_name} completed! Downloading data..."
    #                 )
    #                 # Download the results
    #                 result_file = self.client.files.content(job.output_file_id).content

    #                 with open(f"{self.id_folder}output_fileids.txt", mode="a") as f:
    #                     f.write(f"{job.output_file_id}\t{file_name}\n")

    #                 with open(
    #                     f"{self.batch_output_folder}output_{file_name}", mode="wb"
    #                 ) as f:
    #                     f.write(result_file)

    #                 self.log.info(f"    Downloaded results for file {file_name}")
    #                 self.log.info("    Now sleeping for 5 minutes to ensure that OpenAI's enqueued token count is reset correctly...")
    #                 time.sleep(60*5)
    #                 break

    #             self.log.info(
    #                 f"    Batch job {batch_id} for file {file_name} still in progress. Retrying in 5 minutes..."
    #             )
    #             time.sleep(60*5)

    #     self.log.info("Finished submitting and processing all batch jobs.")

    def submit_batch_jobs(self):
        """
        Submits batch jobs based on file IDs recorded in `self.id_folder`/fileids.txt.
        Records batch IDs locally in `self.id_folder/batchids.txt`
        """
        self.log.info("Submitting batch jobs...")
        file_ids = []
        with open(f"{self.id_folder}fileids.txt", mode="r") as f:
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

        with open(f"{self.id_folder}batchids.txt", mode="w") as f:
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
        with open(f"{self.id_folder}batchids.txt", mode="r") as f:
            for data in f:
                file_entry = data.split("\t")
                batch_ids.append((file_entry[0], file_entry[1].strip()))

        self.log.info(f"Retrieved {len(batch_ids)} batch IDs from local storage")

        # clear out the output_fileids.txt file
        with open(f"{self.id_folder}output_fileids.txt", mode="w") as f:
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

                    with open(f"{self.id_folder}output_fileids.txt", mode="a") as f:
                        f.write(f"{job.output_file_id}\t{file_name}\n")

                    with open(
                        f"{self.batch_output_folder}output_{file_name}", mode="wb"
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
        with open(f"{self.id_folder}fileids.txt", mode="r") as f:
            for data in f:
                file_entry = data.split("\t")
                file_ids.append((file_entry[0], file_entry[1].strip()))
        with open(f"{self.id_folder}output_fileids.txt", mode="r") as f:
            for data in f:
                file_entry = data.split("\t")
                file_ids.append((file_entry[0], "output_" + file_entry[1].strip()))

        self.log.info(f"Retrieved {len(file_ids)} file IDs")

        for file_id, name in file_ids:
            self.log.info(f"Deleting {name} with ID {file_id}")
            self.client.files.delete(file_id)
            time.sleep(2)

        self.log.info("Finished deleting files in OpenAI storage")
