import datasets
import lm_dataformat
from dataclasses import dataclass


class DataDirConfig(datasets.BuilderConfig):
    def __init__(self, data_dir):
        self.data_dir = data_dir


class GPTClippyData(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [DataDirConfig("data")]

    def _info(self):
        self.data_dir = self.config.data_dir
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "repo_name": datasets.Value("string"),
                    "file_name": datasets.Value("string"),
                    "repo_language": datasets.Value("string"),
                    "mime_type": datasets.Value("string"),
                    "stars": datasets.Value("int32"),
                }
            )
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN)]

    def _generate_examples(self, **kwargs):
        rdr = lm_dataformat.Reader(self.data_dir)
        for i, doc in enumerate(rdr.stream_data(get_meta=True)):
            code, metadata = doc
            metadata["text"] = code
            yield i, metadata
