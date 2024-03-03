import laspy
import pandas as pd
import logging
import pathlib

def parse_las_point_cloud(
        file_path: str, output_filename: str, 
        output_folder:str = None, 
        format='hdf5',
        save_to_disk: bool = True
        ):
    logging.basicConfig(level=logging.INFO)

    # parse file path arguments
    folder, filename = parse_input_file(file_path)
    full_file_path = folder / filename

    # validate output folder
    if output_folder is None:
        output_folder = pathlib.Path(__file__).parent
        if str(output_folder).contains("notebooks"):
            output_folder = pathlib.Path(__file__).parent.parent / "data"
        else:
            output_folder = pathlib.Path(__file__).parent / "data"
    validate_output_file(output_folder)
    output_file_path = output_folder / output_filename

    try:
        # read las file
        las = laspy.read(full_file_path)
        # check number of points
        num_points = len(las.points)
        logging.info(f"Processing {num_points} points from {filename}")
        # create dataframe
        point_df = process_las_data(las)
        # save results
        if format == 'hdf5' and save_to_disk:
            save_df_to_hdf5(point_df, output_file_path)
            logging.info(f"Saved parsed data to {output_file_path}")
    
    except Exception as e:
        logging.error(f"Error processing {filename}: {e}")

    return point_df

def save_df_to_hdf5(df: pd.DataFrame, output_file_path: str) -> None:
    df.to_hdf(output_file_path, key='df', mode='w')

def parse_input_file(input_file: str) -> tuple[str, str]:
    folder = pathlib.Path(input_file).parent
    filename = pathlib.Path(input_file).name
    return folder, filename

def validate_output_file(output_folder: pathlib.Path) -> None:
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output folder '{output_folder}' did not exist. Created successfully.")
    elif not output_folder.is_dir():
        raise NotADirectoryError(f"Output path '{output_folder}' is not a directory.")

def process_las_data(las: laspy.file.File) -> pd.DataFrame:
    data = set(zip(
        las['X'] * las.header.scale[0] + las.header.offset[0], 
        las['Y'] * las.header.scale[1] + las.header.offset[1], 
        las['Z'] * las.header.scale[2] + las.header.offset[2], 
        las['red'], 
        las['green'], 
        las['blue'],
    ))
    data_df = pd.DataFrame(list(data), columns = ['x', 'y', 'z', 'r', 'g', 'b'])
    return data_df