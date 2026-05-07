import os
import shutil

# ── CONFIGURE THIS ────────────────────────────────────────────────────────────
SOURCE_DIR = r"/Users/saadfarooq/dataset"   # folder that contains your Train/Test subfolders
# ─────────────────────────────────────────────────────────────────────────────

DEST_TRAIN = "dataset/Train"
DEST_TEST  = "dataset/Test"

# Maps common folder name variations → canonical destination
FOLDER_MAP = {
    "train":    DEST_TRAIN,
    "training": DEST_TRAIN,
    "test":     DEST_TEST,
    "testing":  DEST_TEST,
}

def copy_split(src_split_dir, dest_split_dir):
    """Copy every class subfolder from src_split_dir into dest_split_dir."""
    os.makedirs(dest_split_dir, exist_ok=True)
    class_folders = [f for f in os.listdir(src_split_dir)
                     if os.path.isdir(os.path.join(src_split_dir, f))]

    if not class_folders:
        print(f"  Warning: no subfolders found in {src_split_dir}")
        return

    for class_name in class_folders:
        src_class  = os.path.join(src_split_dir, class_name)
        dest_class = os.path.join(dest_split_dir, class_name)

        if os.path.exists(dest_class):
            shutil.rmtree(dest_class)   # overwrite if already present

        shutil.copytree(src_class, dest_class)
        n = len([f for f in os.listdir(dest_class)
                 if os.path.isfile(os.path.join(dest_class, f))])
        print(f"  Copied class '{class_name}': {n} images → {dest_class}")

def main():
    if not os.path.isdir(SOURCE_DIR):
        print(f"Error: SOURCE_DIR not found:\n  {SOURCE_DIR}")
        print("Edit the SOURCE_DIR variable at the top of this file and try again.")
        return

    top_level = os.listdir(SOURCE_DIR)
    matched = {}
    for name in top_level:
        key = name.lower()
        if key in FOLDER_MAP and os.path.isdir(os.path.join(SOURCE_DIR, name)):
            matched[key] = name

    if "train" not in matched and "training" not in matched:
        print("Error: could not find a Train/Training folder inside SOURCE_DIR.")
        print(f"Found: {top_level}")
        return
    if "test" not in matched and "testing" not in matched:
        print("Error: could not find a Test/Testing folder inside SOURCE_DIR.")
        print(f"Found: {top_level}")
        return

    for key, original_name in matched.items():
        dest = FOLDER_MAP[key]
        src  = os.path.join(SOURCE_DIR, original_name)
        print(f"\nCopying '{original_name}' → {dest}")
        copy_split(src, dest)

    # Summary
    train_classes = [f for f in os.listdir(DEST_TRAIN) if os.path.isdir(os.path.join(DEST_TRAIN, f))]
    test_classes  = [f for f in os.listdir(DEST_TEST)  if os.path.isdir(os.path.join(DEST_TEST,  f))]
    print(f"\nDone.")
    print(f"  dataset/Train — {len(train_classes)} classes: {sorted(train_classes)}")
    print(f"  dataset/Test  — {len(test_classes)}  classes: {sorted(test_classes)}")

if __name__ == "__main__":
    main()
