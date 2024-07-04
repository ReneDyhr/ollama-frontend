import time
from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import requests

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

class RunpodServerlessLLM(LLM):
    pod_id: str
    api_key: str
    request_ids: List[str] = []

    @property
    def _llm_type(self) -> str:
        return "runpod_serverless"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None and self._current_job_id is not None:
            #TODO: handle stop sequence
            ...
        response = self._run_generate_request(prompt)
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"pod_id": self.pod_id}

    def _request_headers(self) -> Mapping[str, str]:
        return {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": self.api_key,
        }

    def _request_url(self) -> str:
        return f"https://api.runpod.ai/v2/{self.pod_id}"


    def _run_generate_request(self, prompt: str) -> str:
        headers = self._request_headers()
        input = {
            "method_name": "generate",
            "input": {"model": "llama3:8b", "prompt": prompt, "system": "You are an assistant to help find the relevant information. You will always provide the source from the context. If you are unable to find the answer from the context, please let me know."},
        }
        # print("before request", input, self._request_url(), headers)
        
        # TODO: Handle network errors
        out = requests.post(
            f"{self._request_url()}/run",
            headers=headers,
            json={"input": input},
        ).json()

        id = out["id"]
        self.request_ids.append(id)

        while out["status"] != "COMPLETED":
            out = requests.get(
                f"{self._request_url()}/status/{id}",
                headers=headers,
            ).json()
            time.sleep(1)

        return out["output"]["response"]