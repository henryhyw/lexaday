import requests
import time
import base64
import sys

# -----------------------------
# Configuration: Update these values!
# -----------------------------
RUNPOD_API_URL = "https://api.runpod.io/graphql"
RUNPOD_API_TOKEN = "rpa_JDV8G48KRP2GDFOB05HRHB1D2L25P2NP9OE5KC2A1v2sxl"  # Your RunPod API token
TEMPLATE_ID = "3vpajpaed4"  # Your RunPod pod template ID (preconfigured with your custom image & volume)
# -----------------------------

# Headers for authentication with RunPod API
HEADERS = {
    "Authorization": f"Bearer {RUNPOD_API_TOKEN}"
}

def deploy_pod():
    """
    Deploys an on-demand pod using RunPod's GraphQL API based on the API reference.
    Adjust the fields if needed.
    """
    query = """
    mutation {
    podFindAndDeployOnDemand(input: {
        cloudType: ALL
        gpuCount: 1
        volumeInGb: 40
        containerDiskInGb: 40
        minVcpuCount: 2
        minMemoryInGb: 15
        gpuTypeId: "NVIDIA RTX 4090"
        name: "Lexaday API"
        dockerArgs: ""
    }) {
        id
        status
    }
    }
    """

    variables = {}  # no variables needed since all values are in the query string
    print("Deploying pod with full input object...")
    response = requests.post(RUNPOD_API_URL, json={"query": query, "variables": variables}, headers=HEADERS)
    print("Deploy response:", response.text)
    if response.status_code != 200:
        sys.exit("Error deploying pod: " + response.text)
    data = response.json()
    try:
        pod_id = data["data"]["podFindAndDeployOnDemand"]["id"]
    except Exception as e:
        sys.exit("Error parsing pod deployment response: " + str(e))
    print(f"Pod deployed with ID: {pod_id}")
    return pod_id

def poll_pod_status(pod_id, timeout=300, interval=10):
    """
    Polls the pod status until it's RUNNING and returns its public IP.
    """
    query = """
    query PodStatus($podId: String!) {
      pod(podId: $podId) {
        id
        status
        publicIp
      }
    }
    """
    variables = {"podId": pod_id}
    print("Waiting for pod to become RUNNING...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        response = requests.post(RUNPOD_API_URL, json={"query": query, "variables": variables}, headers=HEADERS)
        if response.status_code != 200:
            print("Error checking pod status: " + response.text)
            time.sleep(interval)
            continue
        pod_data = response.json()["data"]["pod"]
        status = pod_data["status"]
        public_ip = pod_data.get("publicIp")
        print(f"Pod status: {status}")
        if status == "RUNNING" and public_ip:
            print(f"Pod is running at IP: {public_ip}")
            return public_ip
        time.sleep(interval)
    sys.exit("Pod did not become RUNNING within the timeout period.")

def terminate_pod(pod_id):
    """
    Terminates the pod using the RunPod GraphQL API.
    """
    mutation = """
    mutation TerminatePod($podId: String!) {
      podTerminate(podId: $podId)
    }
    """
    variables = {"podId": pod_id}
    print("Terminating pod...")
    response = requests.post(RUNPOD_API_URL, json={"query": mutation, "variables": variables}, headers=HEADERS)
    if response.status_code == 200:
        print("Pod terminated successfully.")
    else:
        print("Error terminating pod: " + response.text)

def call_generate_text(public_ip, prompt):
    """
    Calls the /generate-text endpoint of your FastAPI server.
    """
    url = f"http://{public_ip}:8000/generate-text"
    payload = {"prompt": prompt}
    print(f"Calling text generation API with prompt: {prompt}")
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()

def call_generate_image(public_ip, prompt):
    """
    Calls the /generate-image endpoint and returns the Base64-encoded image.
    """
    url = f"http://{public_ip}:8000/generate-image"
    payload = {"prompt": prompt}
    print(f"Calling image generation API with prompt: {prompt}")
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()

def main():
    pod_id = None
    try:
        # Step 1: Deploy the pod
        pod_id = deploy_pod()
        
        # Step 2: Poll for pod readiness and get public IP
        public_ip = poll_pod_status(pod_id)
        
        # Step 3: Generate motivational quote using the word "victory"
        text_prompt = "Write a motivational quote including the word 'victory'."
        text_response = call_generate_text(public_ip, text_prompt)
        quote = text_response.get("quote")
        print("\nMotivational Quote:")
        print(quote)
        
        # Step 4: Generate creative prompt using the word "victory"
        creative_prompt_request = "Generate a creative artistic prompt that includes the word 'victory'."
        creative_response = call_generate_text(public_ip, creative_prompt_request)
        creative_prompt = creative_response.get("quote")
        print("\nCreative Prompt:")
        print(creative_prompt)
        
        # Step 5: Generate an image from the creative prompt
        image_response = call_generate_image(public_ip, creative_prompt)
        image_b64 = image_response.get("image_base64")
        if image_b64:
            image_data = base64.b64decode(image_b64)
            image_filename = "generated_image.png"
            with open(image_filename, "wb") as f:
                f.write(image_data)
            print(f"\nImage successfully saved as {image_filename}")
        else:
            print("Image generation failed.")
        
    except Exception as e:
        print("An error occurred during the workflow:")
        print(e)
    finally:
        if pod_id is not None:
            try:
                terminate_pod(pod_id)
            except Exception as e:
                print("Error terminating pod:", e)

if __name__ == "__main__":
    main()