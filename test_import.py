
import sys
import os

# Add the current directory to sys.path so we can import Dfit
sys.path.append(os.getcwd())

try:
    import Dfit
    print(f"Dfit imported: {Dfit}")
    try:
        model = Dfit.Dcov
        print("Dfit.Dcov found")
    except AttributeError:
        print("Dfit.Dcov NOT found")
        try:
            model = Dfit.Dfit.Dcov
            print("Dfit.Dfit.Dcov found")
        except AttributeError:
            print("Dfit.Dfit.Dcov NOT found")

except ImportError as e:
    print(f"ImportError: {e}")
