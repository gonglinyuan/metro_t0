# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

from .dictionary import Dictionary, TruncatedDictionary

from .fairseq_dataset import FairseqDataset, FairseqIterableDataset

from .base_wrapper_dataset import BaseWrapperDataset

from .append_token_dataset import AppendTokenDataset
from .bucket_pad_length_dataset import BucketPadLengthDataset
from .concat_dataset import ConcatDataset
from .concat_capped_dataset import ConcatCappedDataset
from .id_dataset import IdDataset
from .indexed_dataset import (
    IndexedCachedDataset,
    IndexedDataset,
    IndexedRawTextDataset,
    MMapIndexedDataset,
)
from .language_pair_dataset import LanguagePairDataset
from .lm_context_window_dataset import LMContextWindowDataset
from .lru_cache_dataset import LRUCacheDataset
from .mask_tokens_dataset import MaskTokensDataset
from .monolingual_dataset import MonolingualDataset
from .nested_dictionary_dataset import NestedDictionaryDataset
from .numel_dataset import NumelDataset
from .num_samples_dataset import NumSamplesDataset
from .pad_dataset import LeftPadDataset, PadDataset, RightPadDataset, PadShiftDataset
from .prepend_token_dataset import PrependTokenDataset
from .sort_dataset import SortDataset
from .strip_token_dataset import StripTokenDataset
from .t5_dataset import T5Dataset
from .table_lookup_dataset import TableLookupDataset
from .token_block_dataset import TokenBlockDataset
from .shorten_dataset import TruncateDataset, RandomCropDataset

from .iterators import (
    CountingIterator,
    EpochBatchIterator,
    GroupedIterator,
    ShardedIterator,
)

__all__ = [
    "AppendTokenDataset",
    "BaseWrapperDataset",
    "BucketPadLengthDataset",
    "ConcatDataset",
    "ConcatCappedDataset",
    "CountingIterator",
    "Dictionary",
    "EpochBatchIterator",
    "FairseqDataset",
    "FairseqIterableDataset",
    "GroupedIterator",
    "IdDataset",
    "IndexedCachedDataset",
    "IndexedDataset",
    "IndexedRawTextDataset",
    "LanguagePairDataset",
    "LeftPadDataset",
    "LMContextWindowDataset",
    "LRUCacheDataset",
    "MaskTokensDataset",
    "MMapIndexedDataset",
    "MonolingualDataset",
    "NestedDictionaryDataset",
    "NumelDataset",
    "NumSamplesDataset",
    "PadDataset",
    "PadShiftDataset",
    "PrependTokenDataset",
    "RandomCropDataset",
    "RightPadDataset",
    "ShardedIterator",
    "SortDataset",
    "StripTokenDataset",
    "T5Dataset",
    "TableLookupDataset",
    "TokenBlockDataset",
    "TruncateDataset",
    "TruncatedDictionary",
]
