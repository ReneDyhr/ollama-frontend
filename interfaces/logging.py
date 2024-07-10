#!/usr/bin/python
import interfaces.database as db
import uuid
from typing import TypedDict

class LogData(TypedDict):
    id: str
    process_id: str
    start_time: str
    end_time: str
    job: str
    input: str
    output: str

class Logging:
    def __init__(self, start_time: str, end_time: str, job: str, input: str, output: str, process_id: str|None = None):
        if process_id is None:
            process_id = Logging.generate_uuid()
        self.process_id = process_id
        self.start_time = start_time
        self.end_time = end_time
        self.job = job
        self.input = input
        self.output = output

    # get process_id
    def get_process_id(self):
        return self.process_id
    
    # set process_id
    def set_process_id(self, x: str):
        self.process_id = x
        return self
    
    # get start_time
    def get_start_time(self):
        return self.start_time
    
    # set start_time
    def set_start_time(self, x: str):
        self.start_time = x
        return self
    
    # get end_time
    def get_end_time(self):
        return self.end_time
    
    # set end_time
    def set_end_time(self, x: str):
        self.end_time = x
        return self
    
    # get job
    def get_job(self):
        return self.job
    
    # set job
    def set_job(self, x: str):
        self.job = x
        return self
    
    # get input
    def get_input(self):
        return self.input
    
    # set input
    def set_input(self, x: str):
        self.input = x
        return self
    
    # get output
    def get_output(self):
        return self.output
    
    # set output
    def set_output(self, x: str):
        self.output = x
        return self

    @staticmethod
    def generate_uuid():
        return str(uuid.uuid4())
    
    @staticmethod
    def from_id(user_id: int):
        user = db.select_one("SELECT * FROM users WHERE user_id = ?", (user_id,))
        if user is None:
            raise ValueError("User not found")
        return Logging(user["name"], user["email"], user["phone"], user["address"], user["country"], user_id)

    @staticmethod
    def get_all():
        logs = []
        try:
            rows = db.select("SELECT process_id FROM logging GROUP by process_id")

            # convert row objects to dictionary
            for i in rows:
                jobs = db.select("SELECT * FROM logging WHERE process_id = ? ORDER BY id ASC", (i["process_id"],))
                total_time = 0
                job_logs = []
                start_time = 0
                end_time = 0
                cnt = 0
                query = ""
                for job in jobs:
                    cnt += 1
                    if cnt == 1:
                        start_time = job["start_time"]
                        query = job["input"]
                    if cnt == len(jobs):
                        end_time = job["end_time"]
                    # calculate total time
                    total_time += (job["end_time"] - job["start_time"])
                    job_logs.append(Logging.from_dict(job).to_dict())
                logs.append({"process_id": i["process_id"], "query": query, "start_time": start_time, "end_time": end_time, "total_time": total_time, "jobs": job_logs})
        except:
            logs = []

        return logs

    def to_dict(self) -> LogData:
        return {
            "process_id": self.process_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_time": self.end_time - self.start_time,
            "job": self.job,
            "input": self.input,
            "output": self.output
        }

    @staticmethod
    def from_dict(log_dict: LogData):
        return Logging(log_dict["start_time"], log_dict["end_time"], log_dict["job"], log_dict["input"], log_dict["output"], log_dict["process_id"])
    
    # Create user to database
    def create(self):
        log = db.insert("INSERT INTO logging (process_id, start_time, end_time, job, input, output)"
                        " VALUES (?, ?, ?, ?, ?, ?)", (self.process_id, self.start_time, self.end_time, self.job, self.input, self.output));
        self.id = log
        return self

    # Create user table
    def create_db_table():
        success = False
        try:
            conn = db.connect_to_db()
            conn.execute('''
                CREATE TABLE logging (
                    id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                    process_id UUID NOT NULL,
                    start_time DATETIME NOT NULL,
                    end_time DATETIME NOT NULL,
                    job TEXT NOT NULL,
                    input LONGTEXT NOT NULL,
                    output LONGTEXT NOT NULL
                );
            ''')

            conn.commit()
            success = True
        except:
            success = False
        finally:
            conn.close()
        return success