from __future__ import annotations
from time import gmtime, strftime
from collections import namedtuple
import numpy as np
import os
from random import randint, uniform, getrandbits, sample
from librosa import load
from typing import List, Union
import soundfile as sf

from genetic_algorithm import Individual
from utils import log_info
from configuration import TuneGeneratorConf

BLOCKS_DIR_NAME = "piano.mf.wav.15sec"
BLOCKS_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), BLOCKS_DIR_NAME)

Block_attr = namedtuple('Block_attr', ['offset', 'duration', 'amp', 'name'])


class TuneData:
    index_to_block = {}
    block_to_index = {}
    all_blocks = []
    db_initiated = False

    NOTE_NAMES = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7',
                  'Ab1', 'Ab2', 'Ab3', 'Ab4', 'Ab5', 'Ab6', 'Ab7',
                  'B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7',
                  'Bb1', 'Bb2', 'Bb3', 'Bb4', 'Bb5', 'Bb6', 'Bb7',
                  'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8',
                  'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7',
                  'Db1', 'Db2', 'Db3', 'Db4', 'Db5', 'Db6', 'Db7',
                  'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7',
                  'Eb1', 'Eb2', 'Eb3', 'Eb4', 'Eb5', 'Eb6', 'Eb7',
                  'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7',
                  'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7',
                  'Gb1', 'Gb2', 'Gb3', 'Gb4', 'Gb5', 'Gb6']
    BASIC_BLOCKS = {}

    @staticmethod
    def init_db():
        """
        initiates the class data structures, iff that hasn't been done yet.
        """
        if not TuneData.db_initiated:
            # TODO: verify BLOCKS_PATH
            log_info('Started loading notes')
            for i, filename in enumerate(os.listdir(BLOCKS_PATH)):
                block_name = filename.split('.')[0]

                TuneData.all_blocks.append(block_name)
                TuneData.index_to_block[i] = block_name
                TuneData.block_to_index[block_name] = i

            # TODO: load with sr=None to use original sample rate
            TuneData.BASIC_BLOCKS = {name: load(f"{BLOCKS_PATH}/{name}.wav")[0] for name in TuneData.NOTE_NAMES}
            log_info('Finished loading notes')
            TuneData.db_initiated = True

    def __init__(self, data=None):
        if data is None:
            data = []
        self.building_blocks = data
        self.init_db()

    def add_block(self, block: Block_attr):
        self.building_blocks.append(block)

    def remove_block(self, block_to_remove: Block_attr):
        for block in self.building_blocks:
            if block == block_to_remove:
                self.building_blocks.remove(block)
                return

    def get_building_blocks(self):
        return self.building_blocks

    def __len__(self):
        return len(self.building_blocks)


def _sample_block(block_name: str, sample_len: float) -> np.ndarray:
    tune = TuneData.BASIC_BLOCKS[block_name]
    amount_of_samples = int(sample_len * Tune.SAMPLE_RATE)
    return tune[:amount_of_samples]


class Tune(Individual):
    ORIG_TUNE_LEN_SEC = 0
    ORIG_TUNE_LEN_SAMP = 0
    SAMPLE_RATE = 0

    def __init__(self, data: TuneData):
        super(Tune, self).__init__()
        self.data = data

    @staticmethod
    def set_tune_len(tune_len: int):
        Tune.ORIG_TUNE_LEN_SAMP = tune_len
        Tune.ORIG_TUNE_LEN_SEC = Tune.ORIG_TUNE_LEN_SAMP / Tune.SAMPLE_RATE

    @staticmethod
    def set_sr(sr: int):
        Tune.SAMPLE_RATE = sr

    @staticmethod
    def _crossover(data_a: TuneData, data_b: TuneData, cross_point: int) -> Tune:
        """
        Single point crossover between 2 Tunes
        :param data_a: TuneData of first Tune
        :param data_b: TuneData of second Tune
        :param cross_point: index between 0 and len(TuneData)
        :return: Tune with crossover-ed TuneData
        """
        cross_data = TuneData()
        cross_data.building_blocks[:cross_point] = data_a.building_blocks[:cross_point]
        cross_data.building_blocks[cross_point:] = data_b.building_blocks[cross_point:]
        return Tune(cross_data)

    def reproduce_by_single_split(self, other: Tune) -> Tune:
        """
        Reproduces from these two parents by splitting at some point in time and taking a part from each
        :param other: other Tune to reproduce with
        :return: created Tune
        """
        if len(self.data.get_building_blocks()) == 0:
            return other
        if len(other.data.get_building_blocks()) == 0:
            return self

        # sort by offset of blocks
        sorted_a = sorted(self.data.get_building_blocks(), key=lambda x: x.offset)
        sorted_b = sorted(other.data.get_building_blocks(), key=lambda x: x.offset)

        min_offset = min(sorted_a[0].offset, sorted_b[0].offset)
        max_offset = max(sorted_a[-1].offset, sorted_b[-1].offset)

        split_offset = uniform(min_offset, max_offset)

        # Randomly (50%-50%) select from which parent to take each part
        if (bool(getrandbits(1)) and (sorted_a[0].offset < split_offset or split_offset <= sorted_b[-1].offset)) \
                or (sorted_b[0].offset < split_offset or split_offset <= sorted_a[-1].offset):
            offspring_blocks = [block for block in sorted_a if block.offset < split_offset] + \
                               [block for block in sorted_b if split_offset <= block.offset]
        else:
            offspring_blocks = [block for block in sorted_b if block.offset < split_offset] + \
                               [block for block in sorted_a if split_offset <= block.offset]

        offspring_tune_data = TuneData()
        offspring_tune_data.building_blocks = offspring_blocks
        return Tune(offspring_tune_data)

    def reproduce_by_free_random_selection(self, other: Tune) -> Tune:
        """
        Reproduces from these two parents by randomly selecting some blocks from each of them
        :param other: other Tune to reproduce with
        :return: created Tune
        """
        blocks_a = self.data.get_building_blocks()
        blocks_b = other.data.get_building_blocks()

        if len(blocks_a) < len(blocks_b):
            blocks_a, blocks_b = blocks_b, blocks_a

        offspring_size = randint(len(blocks_b), len(blocks_a))

        amount_take_from_short = randint(0, len(blocks_b))

        offspring_blocks = sample(blocks_b, k=amount_take_from_short) + \
                           sample(blocks_a, k=offspring_size - amount_take_from_short)

        offspring_tune_data = TuneData()
        offspring_tune_data.building_blocks = offspring_blocks
        return Tune(offspring_tune_data)

    def original_reproduce(self, other: Tune) -> Tune:
        """
        reproduction from this Tune and another using crossover, taking some blocks from each parent
        :param other: other Tune to reproduce with
        :return: created Tune
        """
        # sort by offset of blocks
        sorted_a = sorted(self.data.get_building_blocks(), key=lambda x: x.offset)
        sorted_b = sorted(other.data.get_building_blocks(), key=lambda x: x.offset)

        # make sure a is the larger list (code assumes this later on)
        if len(sorted_a) < len(sorted_b):
            tmp = sorted_a
            sorted_a = sorted_b
            sorted_b = tmp

        # choose size between size_a and size_b
        size_of_child = randint(len(sorted_b), len(sorted_a))

        # copy a, choose how many and which blocks to take from b, rest will be from a
        child_data = sorted_a.copy()
        amount_to_take = randint(1, len(sorted_b))      # TODO: maybe configure min & max amount to take
        take_from_other = np.random.choice(len(sorted_b), size=amount_to_take, replace=False)

        for i in take_from_other:
            child_data[i] = sorted_b[i]

        # truncate remaining blocks
        child_data = child_data[:size_of_child + 1]

        offspring_data = TuneData()
        offspring_data.building_blocks = child_data
        return Tune(offspring_data)

    def reproduce(self, other: Tune) -> Tune:
        offspring = getattr(self, TuneGeneratorConf.REPRODUCE_FUNC_NAME)(other)

        if len(offspring.data.get_building_blocks()) == 0:
            return self if bool(getrandbits(1)) else other

        return offspring

    def mutate(self):
        """
        Mutation of Tune.
        Randomly selected amount of building blocks to mutate. Randomly select witch blocks will mutate.
        Randomly select values to set to mutated blocks
        :return:
        """
        to_remove = []

        if len(self.data.building_blocks) > 0:
            # on average mutation will be applied to data.size/MUTATION_FRACTION of the blocks
            mu = len(self.data) / TuneGeneratorConf.MUTATION_FRACTION
            sigma = 0.1
            amount_of_mutations = int(np.random.normal(mu, sigma, 1))
            if amount_of_mutations == 0:
                amount_of_mutations = 1
            # chose indices of blocks to mutate
            blocks_to_mutate = np.random.choice(len(self.data.building_blocks), size=amount_of_mutations, replace=False)
            # list of blocks that will be removed - removal only after iterating all mutated block to not change indices
            to_remove = []

            for block_index in blocks_to_mutate:
                # select if to disable note
                if np.random.binomial(1, TuneGeneratorConf.BLOCK_REMOVE_PROB):
                    to_remove.append(self.data.building_blocks[block_index])
                    continue

                # select if to change note
                new_note = self.data.building_blocks[block_index].name
                if np.random.binomial(1, TuneGeneratorConf.NOTE_CHANGE_PROB):
                    # select other note
                    new_note = np.random.choice(TuneData.NOTE_NAMES, size=1)[0]

                # select new offset
                new_offset = uniform(TuneGeneratorConf.MIN_OFFSET, Tune.ORIG_TUNE_LEN_SEC)
                # select new duration
                new_duration = uniform(0, Tune.ORIG_TUNE_LEN_SEC - new_offset)
                # select new amp
                new_amp = uniform(TuneGeneratorConf.MIN_AMP, TuneGeneratorConf.MAX_AMP)

                self.data.building_blocks[block_index] = Block_attr(offset=new_offset, duration=new_duration, amp=new_amp,
                                                                    name=new_note)

        if np.random.binomial(1, TuneGeneratorConf.ADD_BLOCK_PROB):
            amount_to_add = randint(1, TuneGeneratorConf.MAX_BLOCKS_TO_ADD)
            for i in range(amount_to_add):
                new_note = np.random.choice(TuneData.NOTE_NAMES, size=1)[0]
                new_offset = uniform(TuneGeneratorConf.MIN_OFFSET, Tune.ORIG_TUNE_LEN_SEC)
                new_duration = uniform(0, Tune.ORIG_TUNE_LEN_SEC - new_offset)
                new_amp = uniform(TuneGeneratorConf.MIN_AMP, TuneGeneratorConf.MAX_AMP)

                self.data.building_blocks.append(Block_attr(offset=new_offset, duration=new_duration, amp=new_amp,
                                                            name=new_note))

        for b in to_remove:
            self.data.remove_block(b)

    def raw(self):
        """
        :return: np.ndarray of raw data of the Tune
        """
        blocks: Union[List[Block_attr, ...], None]
        block: Block_attr
        raw_data = np.zeros(Tune.ORIG_TUNE_LEN_SAMP)
        for block in self.data.get_building_blocks():
            sample = _sample_block(block.name, Tune.ORIG_TUNE_LEN_SEC - block.offset)
            # sample *= block.amp       # TODO: disabled to overcome bug with values reaching inf causing melspectogram to throw exception
            start_point = int(block.offset * Tune.SAMPLE_RATE)
            raw_data[start_point: start_point + len(sample)] += sample

        return raw_data, Tune.SAMPLE_RATE

    def save_tune(self, name: str, path: str, sr=SAMPLE_RATE, with_timestamp=False):
        if with_timestamp:
            name += ("_" + strftime("%Y-%m-%d__%H_%M_%S", gmtime()))
        if not name.endswith(".wav"):
            name += ".wav"
        out_path = os.path.join(path, name)
        raw_data = self.raw()
        sf.write(out_path, raw_data[0], raw_data[1])


def generate_single_tune():
    # Assuming audio_data is a numpy mono track, sr is the sample rate
    #   and NOTE_DURATION is the max number of seconds a single note can be played
    NOTE_DURATION = 14
    # total_duration_secs = audio_data.size // sr
    # On average about two blocks played per second (with gaussian deviation)
    num_blocks = (int(Tune.ORIG_TUNE_LEN_SEC) * 2) + int(np.random.normal(0, Tune.ORIG_TUNE_LEN_SEC // 4))
    notes = np.random.choice(list(TuneData.BASIC_BLOCKS.keys()), size=num_blocks, replace=True)
    offsets_secs = np.random.randint(0, high=Tune.ORIG_TUNE_LEN_SEC, size=num_blocks)
    durations = np.random.randint((2 * NOTE_DURATION) // 3, high=NOTE_DURATION, size=num_blocks)
    amps = [1] * num_blocks

    tune_data = [Block_attr(*block_data) for block_data in zip(offsets_secs, durations, amps, notes)]
    td = TuneData(tune_data)
    return Tune(td)


def generate_population(size: int) -> List[Tune, ...]:
    return [generate_single_tune() for _ in range(size)]
