# What is this?

I wrote this program to experiment with different version of AI and ML-based image search.

# Prerequisites

1. Launch the vector database:

```
sudo docker compose -f compose.yml up
```

2. Install the required packages:

```
pip install -r requirements.txt
```

# Using this program

For a full list of flags, options, and models, run

```
python run.py --help
```

To index the files in a folder, use ``--ingest_path`` and specify which embedding model to use:

```bash
python run.py --ingest_path <path to files> --model <model name>
```

To run a search, use ``--search_type`` to choose the distance metric and either ``--search_path`` for image-to-image search or ``--search_string`` for text-to-image search. A ``--model`` must also be specified:

```bash
python run.py --search_type cosine --search_path <path> --search_string <string> --model <model name>
```

# Support embeddings

* DinoSmall:
	- Model is facebook/dinov2-small
	- Supports image input
	- Supports cosine, euclid, and innerproduct metrics
* DinoBase:
	- Model is facebook/dinov2-small
	- Supports image input
	- Supports cosine, euclid, and innerproduct metrics
* DinoLarge:
	- Model is facebook/dinov2-small
	- Supports image input
	- Supports cosine, euclid, and innerproduct metrics
* DinoGiant:
	- Model is facebook/dinov2-small
	- Supports image input
	- Supports cosine, euclid, and innerproduct metrics
* ClipVitBasePatch32
	- Model is openai/clip-vit-base-patch32
	- Support both image and text input
	- Supports cosine, euclid, and innerproduct metrics
* PDQHash
	- "Model" is the PDQ Hashing algorithm
	- Supports image input
	- Supports hamming metric
* PHash8
	- "Model" is Perceptual Hash from imagehash with ``hash_size=8``
	- Supports image input
	- Support hamming metric
* PHash16
	- "Model" is Perceptual Hash from imagehash with ``hash_size=16``
	- Supports image input
	- Support hamming metric
