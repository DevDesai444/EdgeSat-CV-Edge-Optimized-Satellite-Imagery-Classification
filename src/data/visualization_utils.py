import argparse
import matplotlib
from pandas.core import indexing
from filter_utils import load_csv_with_file_links, filter_file_list
import fsspec
from matplotlib import pyplot as plt 
import rasterio
from rasterio.plot import show 
import numpy as np 
import wandb
from glob import glob 
from timeit import default_timer as timer

try:
    from src.utils import resolve_wandb_entity, resolve_wandb_mode, maybe_wandb_login
except ImportError:
    from utils import resolve_wandb_entity, resolve_wandb_mode, maybe_wandb_login

try:
    from src.data.gcs_config import bucket_path
except ImportError:
    from gcs_config import bucket_path

# TODO: Maybe also show near infra-red also, good for water ~ will be black prolly
#       (12, 8, 4) ~ good enough info for water vs ground vs vegetation => maybe better visualization
#       or NDWI ~ https://foodsecurity-tep.net/S2_NDWI
# TODO: Add index

def folder_has_images(fs, folder, k = 1):
    files = fs.glob(folder+"/*")
    images = [f for f in files if f[-4:] == ".tif"]
    return len(images) > k

"""
import cv2
def visualize_rgb_fast(tif_path, cut_off_value = 2000, show=False, save="tmp.png"):
    # CV2 or SKIMAGE doesnt read from GS

    #from skimage import io
    
    #src = io.imread("gs://" + tif_path)

    src = rasterio.open("gs://" + tif_path) # Can I open fast in lower resolution?
    print(dir(src))
    
    print(src)
    return None, None #plot, resolution
"""

def visualize_rgb(tif_path, cut_off_value = 2000, show=False, save="tmp.png", force_process_all=False):
    # Hints to speed it up using overviews:
    # - https://rasterio.readthedocs.io/en/latest/topics/overviews.html
    # - https://gis.stackexchange.com/questions/353794/decimated-and-windowed-read-in-rasterio
    # Slow using rasterio ~ 5 images = 41.680561229994055s
    plot = plt.figure()
    
    # Open and read RGB bands
    src = rasterio.open("gs://" + tif_path) # Can I open fast in lower resolution?

    if not force_process_all:
        if src.width*src.height > 3451*4243: #5776040: # 3940,1466 still loaded, larger froze ...
            print("skipping too large~ ", src.width, src.height, src)
            return None, None
        if src.width*src.height < 500*500:
            print("skipping too small~ ", src.width, src.height, src)
            return None, None

    print("opening ~ ", src.width, src.height, src)

    image_rgb = src.read([4,3,2]) # src.read(window=window) would loot at a corner only ...
    red, green, blue = image_rgb

    # Threshold to deal with outliers
    red[red>cut_off_value] = cut_off_value
    blue[blue>cut_off_value] = cut_off_value
    green[green>cut_off_value] = cut_off_value

    # Scale bands
    red = (rasterio.plot.adjust_band(red, kind='linear')*255).astype(np.uint8)
    green = (rasterio.plot.adjust_band(green, kind='linear')*255).astype(np.uint8)
    blue = (rasterio.plot.adjust_band(blue, kind='linear')*255).astype(np.uint8)
    
    array = np.stack([red, green, blue], axis=0) # returns (3, 1497, 1698)
    resolution = array.shape

    rasterio.plot.show(array)
    plot.tight_layout()
    plt.axis('off')

    if show:
        plot.show() # only on non-vm machines
    if save:
        plot.savefig(save)
        
    return plot, resolution

def file_exists(fs, path):
    return fs.exists(path)

def visualize_folder(fs, folder, force_process_all=False):
    csv_with_links = load_csv_with_file_links(fs, folder)
    plots = []
    resolution = 0

    for index, row in csv_with_links.iterrows():
        exists = file_exists(fs, row.filename)
        print(index, row.filename, exists)
        
        if exists:
            plot, resolution = visualize_rgb(row.filename, force_process_all=force_process_all)
            if plot is None and resolution is None: # was too large, skipped ...
                continue
            plots.append(plot)

    return plots, csv_with_links, resolution


def init_wandb():
    maybe_wandb_login(wandb)


def wandb_show_folders(folders, dry_run = False, load_n_rows=15, wandb_table_name = "table_view_v4", force_process_all=False):
    # Note: check more media at https://docs.wandb.ai/guides/track/log
    #       tables at https://docs.wandb.ai/guides/data-vis/log-tables


    if not dry_run:
        init_wandb()
        wandb.init(
            project="dataset_visualization",
            entity=resolve_wandb_entity(),
            mode=resolve_wandb_mode(),
            config={},
        )
        
    data_rows = []
    expected_images = 5
    columns= ["image_"+str(i) for i in range(expected_images)] + ["height", "width", "number_of_images", "folder"]
    try:
        for folder_i, folder in enumerate(folders):
            print(len(data_rows), "/", load_n_rows, "(folder", folder_i,"/444)") 
            if len(data_rows) == load_n_rows: break
            
            if folder_has_images(fs, folder):
                plots, csv_with_links, resolution = visualize_folder(fs, folder, force_process_all=force_process_all)

                number_of_images = len(plots)
                if number_of_images == 0:
                    plt.close('all')
                    continue

                if len(plots) > expected_images:
                    print("We have more than expected images in the folder ... use only the last", expected_images)
                    plots = plots[-expected_images:]

                # as a table:
                images = [wandb.Image(plot) for plot in plots]
                for i in range(expected_images - len(images)):
                    images.append( wandb.Image(np.zeros((100,100,3))) ) # missing images ...

                _, height, width = resolution
                single_row = images + [height, width, number_of_images, folder]
                data_rows.append(single_row)
                plt.close('all')

                # as an entry
                # now represent this as a row in wandb data
                """
                row_dict = {"folder":folder}
                for index, plot in enumerate(plots):
                    # this works, but adds every image as an independent entity
                    # I'd rather have rows ...
                    row_dict["image_"+str(index)] = wandb.Image(plot) 
                    # wandb.log({"example": wandb.Video("myvideo.mp4")})

                    #plot.savefig("image_"+str(index)+".png")
                
                plt.close('all')
                
                if dry_run:
                    print(row_dict)
                else:
                    wandb.log(row_dict)
                """

    except KeyboardInterrupt:
        print("Interupted with", len(data_rows), "samples ... will still try to save them...")

    # Logs only the final table (not row by row)
    print("debug~", data_rows)
    test_table = wandb.Table(data=data_rows, columns=columns)
    wandb.log({wandb_table_name: test_table})

    if not dry_run:
        wandb.finish()




if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='Visualize filtered folders from a GCS-backed dataset and log them to W&B.'
    )
    parser.add_argument(
        '--root-folder',
        default=None,
        help='Explicit gs:// glob for metadata CSVs. Defaults to gs://<EDGESAT_GCS_BUCKET>/train_multiscene/*/S2/*.csv.',
    )
    parser.add_argument('--load-n-rows', type=int, default=None, help='Maximum number of folders to log.')
    parser.add_argument('--dry-run', action='store_true', help='Build the table without starting a W&B run.')
    parser.add_argument('--force-process-all', action='store_true', help='Disable image-size guards during rendering.')
    args = parser.parse_args()

    fs = fsspec.filesystem("gs")
    root_folder = args.root_folder or bucket_path('train_multiscene', '*', 'S2', '*.csv')
    _, filtered_folders, _ = filter_file_list(fs, root_folder)
    print("Prefiltered", len(filtered_folders), "folders.")

    load_n_rows = args.load_n_rows or len(filtered_folders)
    wandb_show_folders(
        filtered_folders,
        dry_run=args.dry_run,
        load_n_rows=load_n_rows,
        force_process_all=args.force_process_all,
    )
