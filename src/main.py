import torch


def main():
    print("🚀 PyTorch sanity check starting...")

    # Check version
    print(f"PyTorch version: {torch.__version__}")

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Available device: {device}")

    # Create simple tensors
    x = torch.rand(2, 3, device=device)
    y = torch.rand(2, 3, device=device)

    # Basic operation
    z = x + y

    print("Tensor x:\n", x)
    print("Tensor y:\n", y)
    print("x + y = \n", z)

    print("✅ PyTorch works!")


if __name__ == "__main__":
    main()
