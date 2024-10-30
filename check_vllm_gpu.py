from vllm import LLM
import torch

def main():
    # Check if PyTorch detects a GPU
    if torch.cuda.is_available():
        print("CUDA is available. GPU is being detected by PyTorch.")
    else:
        print("CUDA is not available. PyTorch is running on CPU.")
    
    # Initialize a simple vLLM model and see if it runs on the GPU
    try:
        llm = LLM(model="gpt2")  # Load a small model for testing purposes
        device = next(llm.model.parameters()).device
        if device.type == 'cuda':
            print("vLLM model is running on GPU.")
        else:
            print("vLLM model is running on CPU.")
    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    main()
