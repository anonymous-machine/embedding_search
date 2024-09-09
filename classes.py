import argparse
import hashlib
import mimetypes
import os

from abc import ABC, abstractmethod
from typing import Optional, Union, Any
from pathlib import Path, PosixPath
from dataclasses import dataclass, field

import numpy as np
import psycopg
import torch
import cv2
import pdqhash
import imagehash

from PIL import Image

from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from transformers import AutoModel, AutoImageProcessor, CLIPModel, CLIPProcessor, AutoProcessor, AutoTokenizer

def init_database() -> None:
	connection = get_db_connection()
	cursor = connection.cursor()
	cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
	connection.commit()
	cursor.close()

def read_in_chunks(file_object, chunk_size=1024):
	while True:
		data = file_object.read(chunk_size)
		if not data:
			break
		yield data

def hash_file(path: Path, chunk_size: int = 65535) -> str:
	hash_fn = hashlib.sha3_256()
	with open(path, "rb") as f:
		for file_chunk in read_in_chunks(f, chunk_size=chunk_size):
			hash_fn.update(file_chunk)
	return str(hash_fn.hexdigest())

def guess_mime_prefix(path) -> str:
	try:
		prefix = mimetypes.guess_type(path)[0].split("/")[0]
	except Exception as e:
		prefix = ""
	return prefix

def get_image_files(path: Path, recursive: bool = False) -> set:
	root = path
	if recursive:
		files = {f for f in root.rglob("*") if guess_mime_prefix(f) == "image"}
	else:
		files = {f for f in root.glob("*") if guess_mime_prefix(f) == "image"}
	return files

def hash_in_table(file_hash: str, search_table) -> bool:
	conn = get_db_connection()
	cursor = conn.cursor()
	search_statement = f"SELECT * FROM {search_table} WHERE file_hash ='{file_hash}' LIMIT 1;"
	cursor.execute(search_statement)
	result = cursor.fetchone()
	if result is None:
		return False
	return True

def cosine_similarity(vector: np.array, table_name: str, search_depth: int, conn: Optional[psycopg.Connection] = None) -> list:
	if conn is None:
		conn = get_db_connection()
	cursor = conn.cursor()
	vector_string = str(vector.tolist())
	search_string = f"SELECT file_path, (1 - (embedding <=> '{vector_string}')) AS similarity FROM {table_name} ORDER BY similarity DESC LIMIT {search_depth};"
	cursor.execute(search_string)
	results = cursor.fetchall()
	return results

def inner_product(vector: np.array, table_name: str, search_depth: int, conn: Optional[psycopg.Connection] = None) -> list:
	if conn is None:
		conn = get_db_connection()
	cursor = conn.cursor()
	vector_string = str(vector.tolist())
	search_string = f"SELECT file_path, (-1 * (embedding <#> '{vector_string}')) AS inner_product FROM {table_name} ORDER BY inner_product DESC LIMIT {search_depth};"
	cursor.execute(search_string)
	results = cursor.fetchall()
	return results

def euclidean_distance(vector: np.array, table_name: str, search_depth: int, conn: Optional[psycopg.Connection] = None) -> list:
	if conn is None:
		conn = get_db_connection()
	cursor = conn.cursor()
	vector_string = str(vector.tolist())
	search_string = f"SELECT file_path, ((embedding <-> '{vector_string}')) AS inner_product FROM {table_name} ORDER BY inner_product ASC LIMIT {search_depth};"
	cursor.execute(search_string)
	results = cursor.fetchall()
	return results

def hamming_distance(vector: np.array, table_name: str, search_depth: int, conn: Optional[psycopg.Connection] = None) -> list:
	if conn is None:
		conn = get_db_connection()
	cursor = conn.cursor()
	vector_string = "".join(vector.astype(str))
	search_string = f"SELECT file_path, ((embedding <~> '{vector_string}')) AS hamming_distance FROM {table_name} ORDER BY hamming_distance ASC LIMIT {search_depth};"
	cursor.execute(search_string)
	results = cursor.fetchall()
	return results

def insert_vector(vector: np.array, table_name: str, file_path: str, file_hash: str, conn: Optional[psycopg.Connection] = None) -> None:
	if conn is None:
		conn = get_db_connection()
	cursor = conn.cursor()
	vector_string = str(vector.tolist())
	insert_statement = f"INSERT INTO {table_name} VALUES (%s, %s, %s);"
	cursor.execute(insert_statement, (file_hash, file_path, vector_string))
	conn.commit()

def insert_bit_vector(vector: np.array, table_name: str, file_path: str, file_hash: str, conn: Optional[psycopg.Connection] = None) -> None:
	if conn is None:
		conn = get_db_connection()
	cursor = conn.cursor()
	vector_string = "".join(vector.astype(str))
	insert_statement = f"INSERT INTO {table_name} VALUES (%s, %s, %s);"
	cursor.execute(insert_statement, (file_hash, file_path, vector_string))
	conn.commit()

def get_db_connection():
	db_host = os.environ.get("POSTGRES_HOST", "localhost")
	db_user = os.environ.get("POSTGRES_USER", "user")
	db_name = os.environ.get("POSTGRES_NAME", "vectordb")
	db_port = os.environ.get("POSTGRES_PORT", "5432")
	if db_port[0] != ":":
		db_port = ":" + db_port
	db_password = os.environ.get("POSTGRES_PASSWORD", "password")
	db_url = f"postgresql://{db_user}:{db_password}@{db_host}{db_port}/{db_name}"
	connection = psycopg.connect(db_url)
	register_vector(connection)
	return connection

@dataclass
class BitEmbeddingModel(ABC):
	model_name: str
	table_name: str
	embedding_size: int
	conn: psycopg.Connection = field(default=None)

	def __post_init__(self):
		self.conn = get_db_connection()
		self.create_table()
		self.initialize()

	@abstractmethod
	def initialize(self, **kwargs):
		pass

	@abstractmethod
	def embed_file(self, path) -> Optional[np.array]:
		pass

	@abstractmethod
	def embed_data(self, data) -> Optional[np.array]:
		pass

	def create_table(self) -> None:
		cursor = self.conn.cursor()
		create_statement = f"""CREATE TABLE IF NOT EXISTS {self.table_name} (file_hash CHAR(64), file_path TEXT, embedding bit({self.embedding_size}));"""
		cursor.execute(create_statement)
		self.conn.commit()

	def ingest_file(self, path: Path, persist:bool=True) -> None:
		file_hash = hash_file(path=path)
		if hash_in_table(file_hash=file_hash, search_table=self.table_name):
			return None
		embedding = self.embed_file(path=path)
		if embedding is None:
			print(f"Error on {path}, skipping")
			return None
		if persist:
			insert_bit_vector(vector=embedding, file_path=str(path), table_name=self.table_name, file_hash=file_hash, conn=self.conn)
		return None

	def hamming_search(self, input_data, search_depth) -> list:
		if type(input_data) is Path:
			vector = self.embed_file(path=input_data)
		else:
			vector = self.embed_data(data=input_data)
		if vector is None:
			return list()
		results = hamming_distance(vector=vector, table_name=self.table_name, search_depth=search_depth, conn=self.conn)
		return results

	def get_all(self):
		cursor = self.conn.cursor()
		cursor.execute(f"SELECT * FROM {self.table_name};")
		results = cursor.fetchall()
		return results


@dataclass
class EmbeddingModel(ABC):
	model_name: str
	table_name: str
	embedding_size: int
	conn: psycopg.Connection = field(default=None)

	def __post_init__(self):
		self.conn = get_db_connection()
		self.create_table()
		self.initialize()

	@abstractmethod
	def initialize(self, **kwargs):
		pass

	@abstractmethod
	def embed_file(self, path) -> Optional[np.array]:
		pass

	@abstractmethod
	def embed_data(self, data) -> Optional[np.array]:
		pass

	def create_table(self) -> None:
		cursor = self.conn.cursor()
		create_statement = f"""CREATE TABLE IF NOT EXISTS {self.table_name} (file_hash CHAR(64), file_path TEXT, embedding vector({self.embedding_size}));"""
		cursor.execute(create_statement)
		self.conn.commit()

	def ingest_file(self, path: Path, persist:bool=True) -> None:
		file_hash = hash_file(path=path)
		if hash_in_table(file_hash=file_hash, search_table=self.table_name):
			return None
		embedding = self.embed_file(path=path)
		if embedding is None:
			print(f"Error on {path}, skipping")
			return None
		if persist:
			insert_vector(vector=embedding, file_path=str(path), table_name=self.table_name, file_hash=file_hash, conn=self.conn)
		return None

	def cosine_search(self, input_data, search_depth) -> list:
		print(f"input_data is {input_data}")
		print(f"type(input_data) is {type(input_data)}")
		if type(input_data) is Path or type(input_data) is PosixPath:
			vector = self.embed_file(path=input_data)
		else:
			vector = self.embed_data(data=input_data)
		if vector is None:
			return list()
		results = cosine_similarity(vector=vector, table_name=self.table_name, search_depth=search_depth, conn=self.conn)
		return results

	def distance_search(self, input_data, search_depth) -> list:
		if type(input_data) is Path:
			vector = self.embed_file(path=input_data)
		else:
			vector = self.embed_data(data=input_data)

		if vector is None:
			return list()
		results = euclidean_distance(vector=vector, table_name=self.table_name, search_depth=search_depth, conn=self.conn)
		return results

	def inner_product_search(self, input_data, search_depth) -> list:
		if type(input_data) is Path:
			vector = self.embed_file(path=input_data)
		else:
			vector = self.embed_data(data=input_data)
		if vector is None:
			return list()
		results = inner_product(vector=vector, table_name=self.table_name, search_depth=search_depth, conn=self.conn)
		return results

	def get_all(self):
		cursor = self.conn.cursor()
		cursor.execute(f"SELECT * FROM {self.table_name};")
		results = cursor.fetchall()
		return results

@dataclass
class DinoSmall(EmbeddingModel):
	model_name:str = "facebook/dinov2-small"
	table_name:str = "facebook_dinov2_small"
	embedding_size: int = 384
	model: AutoModel = field(default=None)
	processor: AutoImageProcessor = field(default=None)

	def initialize(self) -> None:
		device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
		self.model = AutoModel.from_pretrained(self.model_name).to(device)
		self.processor = AutoImageProcessor.from_pretrained(self.model_name)
	
	def embed_data(self, data) -> np.array:
		return self.embed_file(path=data)

	def embed_file(self, path) -> np.array:
		device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

		try:
			img = Image.open(path)
		except Exception as e:
			print(f"Error on {path}: {e}")
			return None

		with torch.no_grad():
			inputs = self.processor(images=img, return_tensors="pt").to(device)
			outputs = self.model(**inputs)
			image_features = outputs.last_hidden_state
			image_features = image_features.mean(dim=1).numpy().reshape(-1)
		return image_features



@dataclass
class DinoBase(EmbeddingModel):
	model_name:str = "facebook/dinov2-base"
	table_name:str = "facebook_dinov2_base"
	embedding_size: int = 384
	model: AutoModel = field(default=None)
	processor: AutoImageProcessor = field(default=None)

	def initialize(self) -> None:
		device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
		self.model = AutoModel.from_pretrained(self.model_name).to(device)
		self.processor = AutoImageProcessor.from_pretrained(self.model_name)
	
	def embed_data(self, data) -> np.array:
		return self.embed_file(path=data)

	def embed_file(self, path) -> np.array:
		device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

		try:
			img = Image.open(path)
		except Exception as e:
			print(f"Error on {path}: {e}")
			return None

		with torch.no_grad():
			inputs = self.processor(images=img, return_tensors="pt").to(device)
			outputs = self.model(**inputs)
			image_features = outputs.last_hidden_state
			image_features = image_features.mean(dim=1).numpy().reshape(-1)
		return image_features



@dataclass
class DinoLarge(EmbeddingModel):
	model_name:str = "facebook/dinov2-large"
	table_name:str = "facebook_dinov2_large"
	embedding_size: int = 384
	model: AutoModel = field(default=None)
	processor: AutoImageProcessor = field(default=None)

	def initialize(self) -> None:
		device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
		self.model = AutoModel.from_pretrained(self.model_name).to(device)
		self.processor = AutoImageProcessor.from_pretrained(self.model_name)
	
	def embed_data(self, data) -> np.array:
		return self.embed_file(path=data)

	def embed_file(self, path) -> np.array:
		device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

		try:
			img = Image.open(path)
		except Exception as e:
			print(f"Error on {path}: {e}")
			return None

		with torch.no_grad():
			inputs = self.processor(images=img, return_tensors="pt").to(device)
			outputs = self.model(**inputs)
			image_features = outputs.last_hidden_state
			image_features = image_features.mean(dim=1).numpy().reshape(-1)
		return image_features



@dataclass
class DinoGiant(EmbeddingModel):
	model_name:str = "facebook/dinov2-giant"
	table_name:str = "facebook_dinov2_giant"
	embedding_size: int = 384
	model: AutoModel = field(default=None)
	processor: AutoImageProcessor = field(default=None)

	def initialize(self) -> None:
		device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
		self.model = AutoModel.from_pretrained(self.model_name).to(device)
		self.processor = AutoImageProcessor.from_pretrained(self.model_name)
	
	def embed_data(self, data) -> np.array:
		return self.embed_file(path=data)

	def embed_file(self, path) -> np.array:
		device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
		try:
			img = Image.open(path)
		except Exception as e:
			print(f"Error on {path}: {e}")
			return None

		with torch.no_grad():
			inputs = self.processor(images=img, return_tensors="pt").to(device)
			outputs = self.model(**inputs)
			image_features = outputs.last_hidden_state
			image_features = image_features.mean(dim=1).numpy().reshape(-1)
		return image_features

@dataclass
class ClipVitBasePatch32(EmbeddingModel):
	model_name:str = "openai/clip-vit-base-patch32"
	table_name:str = "openai_clip_vit_base_patch32"
	embedding_size: int = 512

	def initialize(self) -> None:
		device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
		self.model = CLIPModel.from_pretrained(self.model_name).to(device)
		self.processor = AutoProcessor.from_pretrained(self.model_name)
		self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
	
	def embed_data(self, data) -> np.array:
		inputs = self.tokenizer([data], padding=True,return_tensors="pt")
		text_features = self.model.get_text_features(**inputs).detach().numpy().flatten()
		return text_features

	def embed_file(self, path) -> np.array:
		device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
		try:
			img = Image.open(path)
		except Exception as e:
			print(f"Error on {path}: {e}")
			return None

		with torch.no_grad():
			inputs = self.processor(images=img, return_tensors="pt").to(device)
			image_embedding = self.model.get_image_features(**inputs).detach().cpu().numpy().flatten()
		return image_embedding


@dataclass
class PDQHash(BitEmbeddingModel):
	model_name: str = "pdqhash"
	table_name: str = "pdqhash"
	embedding_size: int = 256

	def initialize(self) -> None:
		pass

	def embed_data(self, data) -> np.array:
		return self.embed_file(path=data)

	def embed_file(self, path) -> np.array:
		image = cv2.imread(path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		hash_vector, quality = pdqhash.compute(image)
		return hash_vector

@dataclass
class PHash8(BitEmbeddingModel):
	model_name: str = "phash8"
	table_name: str = "phash8"
	embedding_size: int= 64

	def initialize(self) -> None:
		pass

	def embed_data(self, data) -> np.array:
		return self.embed_file(path=data)

	def embed_file(self, path) -> np.array:
		hash_size = 8
		image = Image.open(path)
		phash = imagehash.phash(image, hash_size=hash_size)
		phash = str(phash)
		binary = bin(int(phash, 16))[2:]
		padded = binary.zfill(len(phash) * 4)
		array = np.array([int(b) for b in padded])
		return array

@dataclass
class PHash16(BitEmbeddingModel):
	model_name: str = "phash16"
	table_name: str = "phash16"
	embedding_size: int= 256

	def initialize(self) -> None:
		pass

	def embed_data(self, data) -> np.array:
		return self.embed_file(path=data)

	def embed_file(self, path) -> np.array:
		hash_size = 16
		image = Image.open(path)
		phash = imagehash.phash(image, hash_size=hash_size)
		phash = str(phash)
		binary = bin(int(phash, 16))[2:]
		padded = binary.zfill(len(phash) * 4)
		array = np.array([int(b) for b in padded])
		return array
