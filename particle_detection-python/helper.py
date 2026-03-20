import logging
import os


def get_pixel_to_micrometers(resolution: str) -> float:
    if resolution == "4x":
        pixels_to_micrometers = 5e3 / (2530 - 294) # Substitute with your conversion scale
    elif resolution == "2_5x":
        pixels_to_micrometers = 8e3 / (2308 - 85) # Substitute with your conversion scale
    else:
        logging.warning(f"Resolution {resolution} - Unknown reference scale. Using fallback values.")
        pixels_to_micrometers = 1  # Using pixel as unit instead of micrometers
    return pixels_to_micrometers


def extract_and_save_region_properties(csv_folder, image_name, regions, resolution):
    # Can print various parameters for all objects
    # for prop in regions:
    #    logging.info('Label: {} Area: {}'.format(prop.label, prop.area))

    property_list = [
        'area',
        'equivalent_diameter_area',
        'orientation',
        'axis_major_length',
        'axis_minor_length',
        'perimeter',
        'intensity_min',
        'intensity_mean',
        'intensity_max'
    ]

    output_file = open(f"{csv_folder}/{image_name.split('.')[0]}.csv", 'w')
    logging.info(f"Writing to file {output_file}")
    output_file.write(("," + ",".join(property_list) + '\n'))  # writing all the lines with properties of every grain

    pixels_to_micrometers = get_pixel_to_micrometers(resolution)
    for cluster_properties in regions:
        # Output cluster properties to the Excel file
        output_file.write(str(cluster_properties['label']))
        for i, prop in enumerate(property_list):
            if prop == 'area':
                # Convert pixel square to um square
                to_print = cluster_properties[prop] * pixels_to_micrometers ** 2
            elif prop == 'orientation':
                # Convert to degrees from radians
                to_print = cluster_properties[prop] * 57.2958
            elif prop.find('Intensity') < 0:
                # Any prop without Intensity in its name
                to_print = cluster_properties[prop] * pixels_to_micrometers
            else:
                # Remaining props, basically the ones with Intensity in its name
                to_print = cluster_properties[prop]
            output_file.write(',' + str(to_print))
            # logging.info(f"Property {prop}:", to_print)
        output_file.write('\n')

    # Closes the file, otherwise it would be read only.
    output_file.close()


def request_csv_filepath(root_folder: str) -> str:
    """
    Requests which file to select to the user and returns its full path

    :param root_folder: String, folder where to start search
    :return:            String, full filepath
    """
    last_choice = root_folder
    while not last_choice or not last_choice.endswith(".csv"):
        last_choice = request_filepath_to_user(last_choice)
    return last_choice


def request_filepath_to_user(root_folder: str) -> str:
    """
    Requests which file to select to the user and returns its full path

    :param root_folder: String, folder where to start search
    :return:            String, full filepath
    """
    formats: list[str] = [".bias", ".tmp_index", ".raw", ".png"]
    list_of_options: list[str] = [
        option
        for option in os.listdir(root_folder)
        if not any(option.endswith(ext) for ext in formats)
    ]
    user_choice = None
    while not user_choice:
        logging.info("[INDEX]: [ITEM_NAME]")
        for idx, option in enumerate(list_of_options):
            if os.path.isdir(os.path.join(root_folder, option)):
                logging.info(f"({idx + 1}): [{option}]")
            else:
                logging.info(f"({idx + 1}): {option}")
        user_choice = int(input("Choose file to select: ")) or None
    return os.path.join(root_folder, list_of_options[user_choice - 1])