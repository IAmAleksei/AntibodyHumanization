import re
from enum import Enum
from typing import List, Tuple, Optional, Dict

import anarci

from humanization.common import patched_anarci, annotation_const, config_loader
from humanization.common.utils import configure_logger

config = config_loader.Config()
logger = configure_logger(config, "Annotations")


class ChainKind(Enum):
    HEAVY = "H"
    LIGHT = "L"

    def sub_types(self):
        if self == ChainKind.HEAVY:
            return [GeneralChainType.HEAVY]
        elif self == ChainKind.LIGHT:
            return [GeneralChainType.KAPPA, GeneralChainType.LAMBDA]
        else:
            raise RuntimeError("Unrecognized chain kind")


class GeneralChainType(Enum):
    HEAVY = "H"
    KAPPA = "K"
    LAMBDA = "L"

    def specific_type_class(self):
        if self == GeneralChainType.HEAVY:
            return HeavyChainType
        elif self == GeneralChainType.KAPPA:
            return KappaChainType
        elif self == GeneralChainType.LAMBDA:
            return LambdaChainType
        else:
            raise RuntimeError("Unrecognized chain type")

    def specific_type(self, v_type):
        return self.specific_type_class()(str(v_type))

    def kind(self):
        if self == GeneralChainType.HEAVY:
            return ChainKind.HEAVY
        elif self == GeneralChainType.KAPPA or self == GeneralChainType.LAMBDA:
            return ChainKind.LIGHT
        else:
            raise RuntimeError("Unrecognized chain type")

    def available_specific_types(self):
        return [specific_type.value for specific_type in self.specific_type_class()]


class ChainType(Enum):
    @classmethod
    def general_type(cls) -> GeneralChainType:
        ...

    def full_type(self):
        return f"{self.general_type().value}V{self.value}"

    def oas_type(self):
        return f"IG{self.full_type()}"

    @staticmethod
    def from_full_type(s):
        # {H,K,L}V[1-10]
        return GeneralChainType(s[0]).specific_type(s[2:])

    @staticmethod
    def from_oas_type(s):
        # IG{H,K,L}V[1-10]
        return ChainType.from_full_type(s[2:])


class HeavyChainType(ChainType):
    V1 = "1"
    V2 = "2"
    V3 = "3"
    V4 = "4"
    V5 = "5"
    V6 = "6"
    V7 = "7"

    def general_type(self):
        return GeneralChainType.HEAVY


class LightChainType(ChainType):
    pass


class KappaChainType(LightChainType):
    V1 = "1"

    def general_type(self):
        return GeneralChainType.KAPPA


class LambdaChainType(LightChainType):
    V1 = "1"

    def general_type(self):
        return GeneralChainType.LAMBDA


def segments_to_columns(segments: List[Tuple[str, List[str]]]) -> List[str]:
    result = []
    for segment_name, segment_positions in segments:
        for idx, _ in enumerate(segment_positions):
            result.append(f"{segment_name}_{idx + 1}")
    return result


SEGMENTS_ORDER = ["fwr1", "cdr1", "fwr2", "cdr2", "fwr3", "cdr3", "fwr4"]


def compare_positions(first: str, second: str):
    if first[:4] == second[:4]:  # AAAX_YY
        return int(first[5:]) < int(second[5:])
    else:
        return SEGMENTS_ORDER.index(first[:4]) < SEGMENTS_ORDER.index(second[:4])


class Annotation:
    name = "-"
    positions: List[str] = []
    kind: ChainKind = None
    segments: List[Tuple[str, List[str]]] = []
    segmented_positions: List[str] = []
    required_positions: Dict[str, str] = {}
    v_gene_end: int = None


def get_position_segment(positions: List[str], start: str, end: str) -> List[str]:
    def compare(fst, snd):
        fst_ = re.sub(r'\D', '', fst)
        snd_ = re.sub(r'\D', '', snd)
        return len(fst_) < len(snd_) if len(fst_) != len(snd_) else fst < snd

    res = [position for position in positions if not compare(position, start) and compare(position, end)]
    return res


class ChothiaLight(Annotation):
    name = "chothia"
    positions = annotation_const.CHOTHIA_LIGHT_POSITIONS
    kind = ChainKind.LIGHT
    segments = [
        ("fwr1", get_position_segment(positions, "0", "24")),
        ("cdr1", get_position_segment(positions, "24", "35")),
        ("fwr2", get_position_segment(positions, "35", "50")),
        ("cdr2", get_position_segment(positions, "50", "57")),
        ("fwr3", get_position_segment(positions, "57", "89")),
        ("cdr3", get_position_segment(positions, "89", "98")),
        ("fwr4", get_position_segment(positions, "98", "110")),
    ]
    segmented_positions = segments_to_columns(segments)
    required_positions = {'fwr1_24': 'C', 'fwr2_1': 'W', 'fwr3_32': 'C'}
    v_gene_end = segmented_positions.index('fwr3_32')


class ChothiaHeavy(Annotation):
    name = "chothia"
    positions = annotation_const.CHOTHIA_HEAVY_POSITIONS
    kind = ChainKind.HEAVY
    segments = [
        ("fwr1", get_position_segment(positions, "0", "26")),
        ("cdr1", get_position_segment(positions, "26", "33")),
        ("fwr2", get_position_segment(positions, "33", "52")),
        ("cdr2", get_position_segment(positions, "52", "57")),
        ("fwr3", get_position_segment(positions, "57", "95")),
        ("cdr3", get_position_segment(positions, "95", "103")),
        ("fwr4", get_position_segment(positions, "103", "114")),
    ]
    segmented_positions = segments_to_columns(segments)
    required_positions = {'fwr1_23': 'C', 'fwr2_15': 'W', 'fwr3_39': 'C'}
    v_gene_end = segmented_positions.index('fwr3_41')


class Imgt(Annotation):
    name = "imgt"
    positions = annotation_const.IMGT_HEAVY_POSITIONS
    kind = ChainKind.HEAVY
    segments = [
        ("fwr1", get_position_segment(positions, '1', '26')),
        ("cdr1", get_position_segment(positions, '26', '34')),
        ("fwr2", get_position_segment(positions, '34', '51')),
        ("cdr2", get_position_segment(positions, '51', '57')),
        ("fwr3", get_position_segment(positions, '57', '93')),
        ("cdr3", get_position_segment(positions, '93', '103')),
        ("fwr4", get_position_segment(positions, '103', '129')),
    ]
    segmented_positions = segments_to_columns(segments)


def load_annotation(schema: str, kind: ChainKind) -> Annotation:
    if schema == "chothia":
        if kind == ChainKind.HEAVY:
            return ChothiaHeavy()
        elif kind == ChainKind.LIGHT:
            return ChothiaLight()
    elif schema == "imgt":
        if kind == ChainKind.HEAVY:
            return Imgt()
    raise RuntimeError("Unrecognized annotation type")


def annotate_batch(sequences: List[str], annotation: Annotation, chain_type: GeneralChainType = None,
                   is_human: bool = False) -> Tuple[List[int], List[List[str]]]:
    sequences_ = list(enumerate(sequences))
    kwargs = {
        'ncpu': config.get(config_loader.NCPU),
        'scheme': annotation.name,
    }
    if chain_type:
        kwargs['allow'] = {chain_type.value}
    else:
        kwargs['allow'] = {chain_type.value for chain_type in annotation.kind.sub_types()}
    if is_human:
        kwargs['allowed_species'] = ['human']
    else:
        kwargs['allowed_species'] = ['mouse', 'rat', 'rabbit', 'rhesus', 'pig', 'alpaca']
    import sys
    sys.modules['anarci.anarci']._parse_hmmer_query = patched_anarci._parse_hmmer_query  # Monkey patching
    logger.debug(f"Anarci run on {len(sequences)} rows (kwargs: {kwargs})")
    # 'SSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPRFGQGT' fails anarci
    temp_res = anarci.run_anarci(sequences_, **kwargs)
    numerated_sequences = temp_res[1]
    logger.debug(f"Anarci run is finished")
    index_results = []
    prepared_results = []
    for i, numerated_seq in enumerate(numerated_sequences):
        assert sequences_[i][0] == temp_res[0][i][0]
        if numerated_seq is not None:
            a = numerated_seq[0]
            b = a[0]
            seq_dict = {f"{idx}{letter.strip()}": aa for (idx, letter), aa in b if aa != "-"}
            result_seq = [
                seq_dict.get(position, "X")
                for segment_name, segment_positions in annotation.segments for position in segment_positions
            ]
            index_results.append(i)
            prepared_results.append(result_seq)
    logger.debug(f"Anarci returned {len(index_results)} rows")
    return index_results, prepared_results


def annotate_single(sequence: str, annotation: Annotation, chain_type: GeneralChainType) -> Optional[List[str]]:
    _, annotated_seq = annotate_batch([sequence], annotation, chain_type=chain_type)
    if len(annotated_seq) == 1:
        return annotated_seq[0]
    else:
        return None


if __name__ == '__main__':
    annotation = load_annotation("chothia", ChainKind.HEAVY)
    res = annotate_single(
        'EIQLVQSGPELKQPGETVRISCKASGYTFTNYGMNWVKQAPGKGLKWMGWINTYTGEPTYAADFKRRFTFSLETSASTAYLQISNLKNDDTATYFCAKYPHYYGSSHWYFDVWGAGTTVTVSS',
        annotation, GeneralChainType.HEAVY
    )
    print(res)
