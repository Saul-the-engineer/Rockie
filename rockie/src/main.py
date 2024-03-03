from preprocess_las import parse_las_point_cloud

if __name__ == '__main__':
    data = parse_las_point_cloud(
        file_path = "/home/saul/workspace/rockie/rockie/src/raw_data/Flood Monument East.las", 
        output_filename = "tabular_point_cloud.h5", 
        chunk_size=1000000,
        num_processes=4,
        )