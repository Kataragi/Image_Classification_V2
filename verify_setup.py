#!/usr/bin/env python3
"""
Verification script to check if the environment is set up correctly
"""

import sys


def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("   ❌ Python 3.8 or higher is required")
        return False
    else:
        print("   ✅ Python version OK")
        return True


def check_torch():
    """Check PyTorch installation"""
    print("\n🔥 Checking PyTorch...")
    try:
        import torch
        print(f"   PyTorch version: {torch.__version__}")

        if torch.cuda.is_available():
            print(f"   ✅ CUDA is available")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

            # Test GPU operation
            x = torch.randn(100, 100).cuda()
            y = x @ x
            assert y.is_cuda
            print(f"   ✅ GPU operations working")
        else:
            print(f"   ⚠️  CUDA not available - will run on CPU")

        return True
    except ImportError:
        print("   ❌ PyTorch not installed")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def check_torchvision():
    """Check torchvision installation"""
    print("\n👁️  Checking torchvision...")
    try:
        import torchvision
        print(f"   torchvision version: {torchvision.__version__}")
        print("   ✅ torchvision OK")
        return True
    except ImportError:
        print("   ❌ torchvision not installed")
        return False


def check_timm():
    """Check timm installation"""
    print("\n🤖 Checking timm (PyTorch Image Models)...")
    try:
        import timm
        print(f"   timm version: {timm.__version__}")
        print("   ✅ timm OK")
        return True
    except ImportError:
        print("   ❌ timm not installed")
        return False


def check_other_packages():
    """Check other required packages"""
    print("\n📦 Checking other packages...")

    packages = {
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'sklearn': 'scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'tqdm': 'tqdm',
        'tensorboard': 'TensorBoard'
    }

    all_ok = True

    for module_name, display_name in packages.items():
        try:
            if module_name == 'PIL':
                import PIL
                version = PIL.__version__
            elif module_name == 'sklearn':
                import sklearn
                version = sklearn.__version__
            else:
                module = __import__(module_name)
                version = getattr(module, '__version__', 'unknown')

            print(f"   ✅ {display_name}: {version}")
        except ImportError:
            print(f"   ❌ {display_name} not installed")
            all_ok = False

    return all_ok


def check_directories():
    """Check if required directories exist"""
    print("\n📁 Checking directories...")

    import os

    dirs = ['models', 'utils', 'checkpoints', 'logs', 'visualizations', 'dataset']

    all_ok = True

    for dir_name in dirs:
        if os.path.isdir(dir_name):
            print(f"   ✅ {dir_name}/ exists")
        else:
            print(f"   ⚠️  {dir_name}/ not found (will be created if needed)")
            if dir_name == 'dataset':
                all_ok = False

    return all_ok


def check_model_availability():
    """Check if ConvNeXtV2 model is available"""
    print("\n🏗️  Checking ConvNeXtV2 model availability...")

    try:
        import timm
        model_name = 'convnextv2_base.fcmae_ft_in22k_in1k_384'

        # Check if model is in timm
        available_models = timm.list_models('convnextv2*')

        if model_name in available_models:
            print(f"   ✅ {model_name} is available")
            return True
        else:
            print(f"   ⚠️  Model not found in timm, but will be downloaded on first use")
            return True
    except Exception as e:
        print(f"   ❌ Error checking model: {e}")
        return False


def main():
    """Run all checks"""
    print("=" * 60)
    print("🔍 Art Style Classification - Environment Verification")
    print("=" * 60)

    checks = [
        check_python_version(),
        check_torch(),
        check_torchvision(),
        check_timm(),
        check_other_packages(),
        check_directories(),
        check_model_availability()
    ]

    print("\n" + "=" * 60)
    if all(checks):
        print("✅ All checks passed! You're ready to train.")
        print("\n📝 Next steps:")
        print("   1. Prepare your dataset: bash setup_dataset.sh")
        print("   2. Place images in dataset/ folders")
        print("   3. Start training: python train.py --dataset dataset")
    else:
        print("⚠️  Some checks failed. Please install missing dependencies:")
        print("   pip install -r requirements.txt")
    print("=" * 60)


if __name__ == '__main__':
    main()
