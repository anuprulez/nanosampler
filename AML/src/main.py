from preprocess import Preprocess

def main():

    base_path = "nanodiag_datasets/GSE175758/"
    probe_data_path = base_path + "GSE175758_GEO_processed.txt"
    probe_mapper_path = base_path + "GPL13534-11288-mapper-HMBC450.txt"

    # Instantiate Preprocess class
    preprocessor = Preprocess(probe_data_path, probe_mapper_path)

    # Filter and merge data
    preprocessor.merge_data()
    

if __name__ == "__main__":
    main()