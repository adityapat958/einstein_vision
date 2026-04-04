from scipy.io import loadmat
import numpy as np

# fallback import for older SciPy versions
try:
    from scipy.io.matlab import varmats_from_mat
except ImportError:
    from scipy.io.matlab._mio5 import varmats_from_mat

mat_path = "/home/alien/cv_p3/calibration.mat"

def describe(name, obj, level=0):
    indent = "  " * level

    if isinstance(obj, np.ndarray):
        print(f"{indent}{name}: ndarray shape={obj.shape}, dtype={obj.dtype}")

        # likely intrinsic camera matrix
        if obj.shape == (3, 3) and np.issubdtype(obj.dtype, np.number):
            print(f"{indent}>>> POSSIBLE CAMERA MATRIX:")
            print(obj)
            print()

        # matlab struct array
        if obj.dtype.names is not None:
            for field in obj.dtype.names:
                try:
                    describe(f"{name}.{field}", obj[field], level + 1)
                except Exception as e:
                    print(f"{indent}  failed field {field}: {e}")

        # object arrays
        elif obj.dtype == object:
            for idx, item in np.ndenumerate(obj):
                describe(f"{name}{idx}", item, level + 1)

    else:
        print(f"{indent}{name}: {type(obj)} -> {obj}")

print("=== Normal loadmat ===")
try:
    data = loadmat(mat_path)
    for k, v in data.items():
        if not k.startswith("__"):
            describe(k, v)
except Exception as e:
    print("loadmat failed:", e)

print("\n=== Duplicate-variable split ===")
with open(mat_path, "rb") as f:
    entries = varmats_from_mat(f)

print(f"Found {len(entries)} entries")

for i, (name, stream) in enumerate(entries):
    safe_name = name if name else f"unnamed_{i}"
    print(f"\n----- entry {i}: {safe_name} -----")
    try:
        item = loadmat(stream)
        for k, v in item.items():
            if not k.startswith("__"):
                describe(f"{safe_name}.{k}", v)
    except Exception as e:
        print(f"Could not parse entry {i}: {e}")