# import glob
# import re
from pathlib import Path

# import pandas as pd
import pyprojroot

root: Path = pyprojroot.find_root(pyprojroot.has_dir(".devcontainer"))

# # DATASETS

data: Path = root / "icsspy/data"

# british_hansard: Path = data / "british_hansard/split"
# canadian_hansard: Path = data / "canadian_hansard/lipad"
enron: Path = data / "enron"
# copenhagen_networks_study: Path = data / "copenhagen_networks_study"


# def load_data(file: str) -> pd.DataFrame:
#     return pd.read_csv(data / f"{file}.csv")


# def load_hansard(years=None, british=False, canadian=False):
#     # Check for conflicting options
#     if british and canadian:
#         raise ValueError("Both 'british' and 'canadian' cannot be True simultaneously.")

#     # Determine which dataset to load
#     if british:
#         hansard_prefix = "british_hansard"
#         hansard_path = british_hansard
#     elif canadian:
#         hansard_prefix = "canadian_hansard"
#         hansard_path = canadian_hansard
#     else:
#         raise ValueError(
#             "Please specify which Hansard to load: british=True OR canadian=True."
#         )

#     # Flatten the years list if it includes ranges
#     flat_years = []
#     if years is not None:
#         for year in years:
#             if isinstance(year, int):
#                 flat_years.append(year)
#             elif isinstance(year, str) and "-" in year:
#                 start, end = map(int, year.split("-"))
#                 flat_years.extend(range(start, end + 1))
#             else:
#                 raise ValueError(f"Invalid year format: {year}")

#     files_to_load = []
#     if canadian:
#         for year_dir in hansard_path.iterdir():
#             if (
#                 year_dir.is_dir()
#                 and year_dir.name.isdigit()
#                 and (int(year_dir.name) in flat_years or not flat_years)
#             ):
#                 for month_dir in year_dir.iterdir():
#                     if month_dir.is_dir():
#                         files_to_load.extend(month_dir.glob("*.csv"))
#     else:  # British Hansard
#         pattern = re.compile(hansard_prefix + r"_(\d{4})\.csv")
#         all_files = glob.glob(str(hansard_path / f"{hansard_prefix}_*.csv"))
#         if years is None:
#             files_to_load = all_files
#         else:
#             files_to_load = [
#                 file
#                 for file in all_files
#                 if int(pattern.search(file).group(1)) in flat_years
#             ]

#     # Load and concatenate the files into a single DataFrame
#     df_list = [pd.read_csv(file, low_memory=False) for file in files_to_load]
#     combined_df = pd.concat(df_list, ignore_index=True)

#     # Reset the index and return the combined DataFrame
#     combined_df.reset_index(drop=True, inplace=True)
#     return combined_df


# # SLIDES AND NOTEBOOKS

# slides_qmd: Path = root / "slides"
# slides_html: Path = root / "docs"
# course_materials: Path = root / "notebooks"
# day1: Path = root / "notebooks/1-introduction"
# day2: Path = root / "notebooks/2-obtaining-data"
# day3: Path = root / "notebooks/3-computational-text-analysis"
# day4: Path = root / "notebooks/4-computational-network-analysis"
# day5: Path = root / "notebooks/5-simulation-abms"
# day6: Path = root / "notebooks/6-project"
# cm_export: Path = root / "notebooks/_export_"
