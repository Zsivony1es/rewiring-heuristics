import logging

from graphgen.SparseDatasetFileGenerator import SparseDatasetFileGenerator


logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

def main():
    generator = SparseDatasetFileGenerator(graph_count=150)
    generator.generate_dataset_files('Source/res/datasets/sparse_dset/raw_dset_list.pt')
    print("Graph generation done!")


if __name__ == '__main__':
    main()