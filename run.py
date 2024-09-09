import argparse
import hashlib
import mimetypes
import os

from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

from classes import DinoSmall, DinoLarge, DinoBase, DinoGiant, PDQHash, PHash8, PHash16, ClipVitBasePatch32, get_image_files, init_database, get_db_connection

def cli():
	parser = argparse.ArgumentParser()
	parser.add_argument("--ingest_path", type=Path, help="Path to either a directory or an image, to be indexed and added to the database.")
	parser.add_argument("--search_path", type=Path, help="Path to image to use for search in image-type searches e.g. DINOv2.")
	parser.add_argument("--search_string", type=str, help="String to use for search in string-type searches e.g. CLIP.")
	parser.add_argument("--search_type", choices={"euclid", "innerproduct", "cosine", "hamming"}, default="cosine", help="Metric to use for search. Hamming search is supported iff the embedding model is a binary embedding.")
	parser.add_argument("--model", choices={"DinoSmall", "DinoBase", "DinoGiant", "DinoLarge", "PDQHash", "PHash8", "PHash16", "ClipVitBasePatch32"}, help="Model to use for embedding and search", default="DinoSmall")
	parser.add_argument("--search_depth", type=int, help="How many result to return from a search.", default=10)
	args = parser.parse_args()
	ingest_path = args.ingest_path
	search_path = args.search_path
	search_string = args.search_string
	search_type = args.search_type
	search_depth = args.search_depth

	match args.model:
		case "DinoSmall":
			model = DinoSmall()
		case "DinoBase":
			model = DinoBase()
		case "DinoLarge":
			model = DinoLarge()
		case "DinoGiant":
			model = DinoGiant()
		case "PDQHash":
			model = PDQHash()
		case "PHash8":
			model = PHash8()
		case "PHash16":
			model = PHash16()
		case "ClipVitBasePatch32":
			model = ClipVitBasePatch32()
		case _:
			raise ValueError("Model name not recognized.")

	images_to_ingest = get_image_files(ingest_path, recursive=True)

	for image in tqdm(images_to_ingest):
		try:
			model.ingest_file(image)
		except Exception as e:
			print(f"Error on {image}: {e}")
			model.conn = get_db_connection()

	if search_path is not None:
		print(f"Running path search")
		if search_type == "cosine":
			path_results = model.cosine_search(input_data=search_path, search_depth=search_depth)
		elif search_type == "euclid":
			path_results = model.distance_search(input_data=search_path, search_depth=search_depth)
		elif search_type == "innerproduct":
			path_results = model.inner_product_search(input_data=search_path, search_depth=search_depth)
		elif search_type == "hamming":
			path_results = model.hamming_search(input_data=search_path, search_depth=search_depth)
		else:
			path_results = list()

		for i, r in enumerate(path_results):
			print(f"{i}: {r[0]} ({r[1]})")

	if search_string is not None:
		print(f"Running string search")
		if search_type == "cosine":
			string_results = model.cosine_search(input_data=search_string, search_depth=search_depth)
		elif search_type == "euclid":
			string_results = model.distance_search(input_data=search_string, search_depth=search_depth)
		elif search_type == "innerproduct":
			string_results = model.inner_product_search(input_data=search_string, search_depth=search_depth)
		elif search_type == "hamming":
			string_results = model.hamming_search(input_data=search_string, search_depth=search_depth)
		else:
			string_results = list()

		for i, r in enumerate(string_results):
			print(f"{i}: {r[0]} ({r[1]})")

		#print(f"all_data is {model.get_all()}")

if __name__ == '__main__':
	load_dotenv()
	init_database()
	cli()
