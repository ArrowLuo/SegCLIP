import os
import pickle

if __name__ == "__main__":
    input_root = "data"
    output_root = "data"

    imgs_dict = {}
    for file_path in range(5):
        file_path = os.path.join(input_root, "{}.pkl".format(file_path))
        with open(file_path, 'rb') as f:
            img_data = pickle.load(f)
        imgs_dict = dict(imgs_dict, **img_data)
        print("Total images: {}".format(len(imgs_dict)))
    # For pickle
    save_path = os.path.join(output_root, "cc3m_train.pkl")
    pickle.dump(imgs_dict, open(save_path, "wb"))
    print("The file is saved to: {}".format(save_path))