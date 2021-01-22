# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import sys
import glob
import argparse
import pickle
import os
from os.path import join, dirname, isdir, isfile, realpath, normpath, abspath, exists
from scipy import interp
import SimpleITK as sitk
import cv2
import json
import shutil
from six import string_types
from collections import OrderedDict
from scipy.ndimage import gaussian_filter1d


def cv2_imwrite(path, img):
    if sys.version_info < (3, 0):
        cv2.imwrite(path.encode("utf-8"), img)
    else:
        cv2.imwrite(path, img)


def convert2unicode(pystr, encoding="utf-8"):
    if sys.version_info < (3, 0):
        if isinstance(pystr, str):
            pystr = pystr.decode(encoding)
    else:
        if isinstance(pystr, bytes):
            pystr = pystr.decode()
    return pystr


try:
    this_filename = __file__
    this_dir = dirname(realpath(this_filename))
except NameError:
    import sys

    this_filename = sys.argv[0]
    if sys.platform in ["win32", "win64"]:
        this_dir = join(dirname(realpath(this_filename)), "bone_age\\ba_utils")
    else:
        this_dir = join(dirname(realpath(this_filename)), "bone_age/ba_utils")


def save_pickle(data, fname, verbose=False):
    if sys.version_info < (3, 0):
        fmt = "w"
    else:
        fmt = "wb"
    with open(fname, fmt) as f:
        if verbose:
            print("saving pickle @ %s" % (fname))
        pickle.dump(data, f)


def load_pickle(fname, encoding="latin1", verbose=False):
    """
    In py3, to load file originally pickled using py2, set  encoding="latin1"
    """
    if not isfile(fname):
        print("loading pickle failed: not a file:", fname)
        raise ValueError
    if verbose:
        print("loading pickle @ %s" % fname)

    if sys.version_info < (3, 0):
        fmt = "r"
        result = pickle.load(open(fname, fmt))
    else:
        fmt = "rb"
        result = pickle.load(open(fname, fmt), encoding=encoding)
    return result


def is_equal_file(
    src_f, dst_f, ext_list=["py", "c", "h", "cpp", "pyx", "sh", "cu", "yaml"]
):
    if not isfile(dst_f):
        return False
    if src_f.rsplit(".", 1)[-1] in ext_list and dst_f.rsplit(".", 1)[-1] in ext_list:
        src_str = open(src_f).read()
        dst_str = open(dst_f).read()
        if src_str == dst_str:
            return True

    return False


def copy_file(
    src_f,
    dst_f,
    symlink=False,
    force_overwrite=True,
    skip_if_equal=True,
    observe_only=False,
    verbose=True,
):
    if skip_if_equal:
        if is_equal_file(src_f, dst_f):
            if verbose:
                print("[skip] " + dst_f)
            return
    if not isdir(dirname(dst_f)):
        os.makedirs(dirname(dst_f))

    if not symlink:
        if observe_only or verbose:
            print("copying => " + dst_f)
        if not observe_only:
            shutil.copyfile(src_f, dst_f)
    else:
        if observe_only or verbose:
            print("linking => " + dst_f)
        if not observe_only:
            if isfile(dst_f) and force_overwrite:
                os.unlink(dst_f)
            os.symlink(src_f, dst_f)


def copy_by_ext(
    src_top,
    dst_top,
    ext_list=["py", "pickle", "xlsx", ".sh", ".c", ".h", ".cu"],
    ignore_list=["py~"],
    rm_dst_first=False,
    skip_if_equal=True,
    exclude_dirs=["Outputs", "log", "cfg", "configs_bak", "tmp"],
    reverse=False,
    observe_only=False,
    fname_filter=None,
    followlinks=True,
    verbose=False,
):
    ignore_file_list = [
        "raise_exp.py",
        "clean.sh",
        "web_main.py",
        "web_shutdown.sh",
        "web_startup.sh",
        "cython_bbox.c",
        "cython_nms.c",
        "bone_age_predictor.py",
    ]

    if reverse:
        src_top, dst_top = dst_top, src_top
    src_top = normpath(src_top)
    dst_top = normpath(dst_top)
    src_file_list = []
    print("src top: %s" % (src_top))
    for dirpath, dirnames, filenames in os.walk(
        src_top, topdown=True, followlinks=followlinks
    ):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for bname in filenames:
            fname = join(dirpath, bname)
            if bname in ignore_file_list:
                print("-- ignore %s" % fname)
            elif isfile(fname):
                ext = fname.rsplit(".", 1)[-1]
                if ext in ext_list and ext not in ignore_list:
                    if fname_filter is not None and fname_filter not in fname:
                        continue
                    src_file_list.append(fname)
                    # print('src: ' + fname)
    # return src_file_list
    print("Number of src files: %d" % (len(src_file_list)))
    dst_file_list = [fpath.replace(src_top, dst_top) for fpath in src_file_list]
    if rm_dst_first:
        if isdir(dst_top):
            shutil.rmtree(dst_top)
            print("rmtree %s" % (dst_top))

    for i in range(len(src_file_list)):
        copy_file(
            src_file_list[i],
            dst_file_list[i],
            symlink=False,
            skip_if_equal=skip_if_equal,
            observe_only=observe_only,
            verbose=verbose,
        )


def float_array_2_str(float_array, decimal=1):
    s = "["
    for i, v in enumerate(float_array):
        if i != 0:
            s += ", "
        v = round(v, decimal)
        v_str = "%4.1f" % v
        s += v_str
    s += "]"
    return s


def is_valid_date(date_str):
    if isinstance(date_str, string_types):
        date_str = str(date_str)
    else:
        return False
    date_str = date_str.strip()
    if len(date_str) != 8:
        return False
    if not date_str.isdigit():
        return False
    return True


def get_ymd(date_str):
    date_str = date_str.strip()
    if len(date_str) != 8:
        return -1, -1, -1
    y, m, d = date_str[:4], date_str[4:6], date_str[6:]
    return int(y), int(m), int(d)


def diff_date(date1, date2):
    """
    date1, date2: 'YYYYMMDD'

    return a float representing 'date2 - date1' in year
    """
    y1, m1, d1 = get_ymd(date1)
    y2, m2, d2 = get_ymd(date2)

    return (y2 - y1) + (m2 - m1) / 12.0 + (d2 - d1) / 365.0


def get_age_from_dcm_info(dcm_info, use_patient_age=False):
    warn_msg = ""
    if use_patient_age:
        patient_age_str = dcm_info["patient_age"]
        if patient_age_str[-1] not in ["D", "W", "M", "Y"]:
            warn_msg = "warning.patient_age_format_invalid"
            return 0.0, warn_msg
        if not patient_age_str[:-1].isdigit():
            warn_msg = "warning.patient_age_not_digit"
            return 0.0, warn_msg
        if patient_age_str[-1] == "D":
            return float(patient_age_str[:-1]) / 365.0
        elif patient_age_str[-1] == "W":
            return float(patient_age_str[:-1]) * 7 / 365.0
        elif patient_age_str[-1] == "M":
            return float(patient_age_str[:-1]) / 12.0
        elif patient_age_str[-1] == "Y":
            return float(patient_age_str[:-1])
        else:
            warn_msg = "warning.patient_age_format_unknown"
            return 0.0, warn_msg
    # use study_date - birth_date
    birth_date_str, study_date_str, series_date_str = (
        dcm_info["birth_date"],
        dcm_info["study_date"],
        dcm_info["series_date"],
    )
    if not is_valid_date(birth_date_str):
        warn_msg = "W_BONEAGE_003"  #'warning.birth_date_missing'
        return 0.0, warn_msg
    if (not is_valid_date(study_date_str)) and (not is_valid_date(series_date_str)):
        warn_msg = "W_BONEAGE_004"  #'warning.series_date_study_date_missing'
        return 0.0, warn_msg

    if study_date_str != None:
        image_date_str = study_date_str
    else:
        image_date_str = series_date_str

    return diff_date(birth_date_str, image_date_str), warn_msg  # year age


bone_ch2en = {
    "桡骨": "radius",
    "尺骨": "ulna",
    "第一掌骨": "metacarpal_1",
    "第三掌骨": "metacarpal_3",
    "第五掌骨": "metacarpal_5",
    "第一近节指骨": "proximal_phalanges_1",
    "第三近节指骨": "proximal_phalanges_3",
    "第五近节指骨": "proximal_phalanges_5",
    "第三中节指骨": "middle_phalanges_3",
    "第五中节指骨": "middle_phalanges_5",
    "第一远节指骨": "distal_phalanges_1",
    "第三远节指骨": "distal_phalanges_3",
    "第五远节指骨": "distal_phalanges_5",
    "头状骨": "capitate",
    "钩骨": "hamate",
    "三角骨": "triquetral",
    "月骨": "lunate",
    "舟骨": "scaphoid",
    "大多角骨": "trapezium",
    "小多角骨": "trapezoid",
}

bone_names_13 = [
    "第五远节指骨",
    "第五中节指骨",
    "第五近节指骨",
    "第五掌骨",
    "第三远节指骨",
    "第三中节指骨",
    "第三近节指骨",
    "第三掌骨",
    "第一远节指骨",
    "第一近节指骨",
    "第一掌骨",
    "尺骨",
    "桡骨",
]

bone_names_20 = [
    "桡骨",
    "尺骨",
    "第一掌骨",
    "第三掌骨",
    "第五掌骨",
    "第一近节指骨",
    "第三近节指骨",
    "第五近节指骨",
    "第三中节指骨",
    "第五中节指骨",
    "第一远节指骨",
    "第三远节指骨",
    "第五远节指骨",
    "头状骨",
    "钩骨",
    "三角骨",
    "月骨",
    "舟骨",
    "大多角骨",
    "小多角骨",
]

bone_names_dict = {
    "rus": [
        "第五远节指骨",
        "第五中节指骨",
        "第五近节指骨",
        "第五掌骨",
        "第三远节指骨",
        "第三中节指骨",
        "第三近节指骨",
        "第三掌骨",
        "第一远节指骨",
        "第一近节指骨",
        "第一掌骨",
        "尺骨",
        "桡骨",
    ],
    "carpal": ["头状骨", "钩骨", "三角骨", "月骨", "舟骨", "大多角骨", "小多角骨"],
}

# area -> ID -> bone_name
bone_names_order_dict = {
    "rus": dict(zip(range(len(bone_names_dict["rus"])), bone_names_dict["rus"])),
    "carpal": dict(
        zip(range(len(bone_names_dict["carpal"])), bone_names_dict["carpal"])
    ),
}


def gen_en_ratings_from_degree_list(degree_list, area="rus"):
    bone_names = bone_names_dict[area]
    assert len(degree_list) == len(bone_names)
    ratings = {}
    for i, bone_name in enumerate(bone_names):
        en_name = bone_ch2en[bone_name]
        ratings[en_name] = degree_list[i]
    return ratings


def get_tw3_age(area, sex, tw3_score):
    tw3_score2age_tab = score2age[area]["tw3"][sex]
    score_range = tw3_score2age_tab.keys()
    min_score, max_score = min(score_range), max(score_range)
    if tw3_score < min_score:
        tw3_score_truncated = min_score  # TODO: Low Age Standard
    elif tw3_score > max_score:
        tw3_score_truncated = max_score  # TODO: High Age Standard
    else:
        tw3_score_truncated = tw3_score
    tw3_age = tw3_score2age_tab[tw3_score_truncated]
    return tw3_age


# def get_ch05_age(area, sex, ch05_score, version='interp', underflow='truncate'): # deprecated
def get_ch05_age(
    area, sex, ch05_score, version="class3", underflow="zero"
):  # class3 clinical trial standard
    """
    version:
      interp (table from shanbaoping, undefined scores interpolated)
      class3 (table from jianglongzhou, all int scores (above low bound) defined & rounded by nearest month)

    underflow:
      truncate  use low bound of score table
      zero      return 0 BA
    """
    ch05_score2age_tab = ch05_score2age_map[version][area][sex]
    score_range = ch05_score2age_tab.keys()
    min_score, max_score = min(score_range), max(score_range)
    if ch05_score < min_score:
        if underflow == "truncate":
            ch05_score_truncated = min_score
        elif underflow == "zero":
            ch05_score_truncated = 0
            return 0.0
        else:
            assert False
    elif ch05_score > max_score:
        ch05_score_truncated = max_score
        assert False, "Value Error, score more than 1000: %d" % (int(ch05_score))
    else:
        ch05_score_truncated = ch05_score
    ch05_age = ch05_score2age_tab[ch05_score_truncated]
    return ch05_age


def get_age_from_tw3_degree_sex(tw3_degree_list, sex, area="rus"):
    """
    tw3_degree_list: keypoint order
    """
    sex = sex.strip()
    if sex == "F":
        sex = "female"
    elif sex == "M":
        sex = "male"
    else:
        assert sex in ["female", "male"], (
            'invalid sex char, expecting F/M, got "%s"' % sex
        )

    order = bone_names_order_dict[area]
    tw3_score_list = []
    for i, tw3_degree in enumerate(tw3_degree_list):
        bone_kind = order[i]
        if isinstance(tw3_degree, str):
            tw3_degree = ord(tw3_degree) - ord("A")
        assert isinstance(tw3_degree, int)
        tw3_score = degree2score[area]["tw3"][sex][bone_kind][tw3_degree]
        tw3_score_list.append(tw3_score)
    # total score
    tw3_scores = np.sum(np.array(tw3_score_list))
    # tw3 score->age
    tw3_age = get_tw3_age(area, sex, tw3_scores)

    return tw3_scores, tw3_age


def get_age_from_degree_sex_core(degree_list, sex, area="rus"):
    """
    area = 'rus':     keypoint crop list order
    area = 'carpal':  standard order
    """
    sex = sex.strip()
    if sex == "F":
        sex = "female"
    elif sex == "M":
        sex = "male"
    else:
        assert 0, 'invalid sex char, expecting F/M, got "%s"' % sex
    bone_names = bone_names_dict[area]
    order = bone_names_order_dict[area]
    chn_degree_int_list = []
    chn_score_list = []
    tw3_degree_list = []
    tw3_score_list = []
    for j, deg in enumerate(degree_list):
        bone_kind = order[j]
        chn_deg_int = int(round(deg))
        if chn_deg_int < 0:
            chn_deg_int = 0  # 未见骨化中心
        if area == "rus":
            chn_bone_score_tab = degree2score[area]["chn"][sex][bone_kind]
            if chn_deg_int >= len(chn_bone_score_tab):
                chn_deg_int = len(chn_bone_score_tab) - 1  # 找到最大合法等级
            while np.isnan(chn_bone_score_tab[chn_deg_int]):
                chn_deg_int -= 1
            chn_degree_int_list.append(chn_deg_int)
            chn_score = chn_bone_score_tab[chn_deg_int]
            chn_score_list.append(chn_score)
        if chn_deg_int == 0:
            chn_deg_int = "未见骨化中心"
        tw3_degree = chn2tw3[area][bone_kind][chn_deg_int]
        if isinstance(tw3_degree, str):
            tw3_degree = ord(tw3_degree) - ord("A")
        tw3_degree_list.append(tw3_degree)

    tw3_scores, tw3_age = get_age_from_tw3_degree_sex(tw3_degree_list, sex, area)
    # ch05 score->age
    if area == "carpal":
        chn_scores = tw3_scores  # CH05 TW3-C Carpal just copy TW3-Carpal scores
    else:
        chn_scores = np.sum(np.array(chn_score_list))
    chn_age = get_ch05_age(area, sex, chn_scores)
    return chn_scores, chn_age, tw3_degree_list, tw3_scores, tw3_age


def calc_sms_ba(degree_list, sex, area, baa_method):
    """
    This function ROUND degree_list TO INT

    baa_method:
      'tw3'  input     tw3_degree_list
             return    tw3_scores, tw3_age

      'chn'  input     chn_degree_list
             return    chn_scores, chn_age
    """
    assert baa_method in ["tw3", "chn"]
    if baa_method == "tw3":
        return get_age_from_tw3_degree_sex(degree_list, sex=sex, area=area)
    else:
        (
            chn_scores,
            chn_age,
            tw3_degree_list,
            tw3_scores,
            tw3_age,
        ) = get_age_from_degree_sex_core(degree_list, sex=sex, area=area)
        return chn_scores, chn_age


def get_age_from_degree_sex(degree_list, sex, area="rus", mean=True):
    """
    sex:
        M for male
        F for female
    area:
        'rus':     in keypoint crop list order
        'carpal':  in standard order

    mean:
        True:      Given multiple annotations, return mean values
    """
    ch05_sms_list = []
    ch05_ba_list = []
    tw3_degs_list = []
    tw3_sms_list = []
    tw3_ba_list = []
    degree_np = np.array(degree_list)
    if degree_np.ndim == 1:
        degree_np = degree_np[np.newaxis, :]
    for i in range(degree_np.shape[0]):
        ch05_sms, ch05_ba, tw3_degs, tw3_sms, tw3_ba = get_age_from_degree_sex_core(
            degree_np[i], sex=sex, area=area
        )
        ch05_sms_list.append(ch05_sms)
        ch05_ba_list.append(ch05_ba)
        tw3_degs_list.append(tw3_degs)
        tw3_sms_list.append(tw3_sms)
        tw3_ba_list.append(tw3_ba)

    ch05_sms_np = np.array(ch05_sms_list)
    ch05_ba_np = np.array(ch05_ba_list)
    tw3_degs_np = np.array(tw3_degs_list)
    tw3_sms_np = np.array(tw3_sms_list)
    tw3_ba_np = np.array(tw3_ba_list)

    if mean:
        ch05_sms_np = ch05_sms_np.mean(axis=0)
        ch05_ba_np = ch05_ba_np.mean(axis=0)
        tw3_degs_np = tw3_degs_np.mean(axis=0)
        tw3_sms_np = tw3_sms_np.mean(axis=0)
        tw3_ba_np = tw3_ba_np.mean(axis=0)

    return ch05_sms_np, ch05_ba_np, tw3_degs_np, tw3_sms_np, tw3_ba_np


def age_flt2str(age_flt, chn=False, decimals=0):
    if np.isnan(age_flt):
        if chn:
            return "-岁-月"
        else:
            return "-Y-M"
    age_flt = float(age_flt)
    y = int(age_flt)
    if decimals == 0:
        m = int(round((age_flt - y) * 12))
    else:
        m = round((age_flt - y) * 12, decimals)
    if chn:
        return "%d岁%s月" % (y, str(m))
    else:
        return "%dY%sM" % (y, str(m))


def age_str2flt(age_str):
    """
    Supported format
    3Y4M
    3y4m
    3岁4月
    3年4月
    4月
    4M
    assert age_str2flt('3岁3月') == 3.25
    assert age_str2flt('3岁0月') == 3.0
    assert age_str2flt('0月') == 0.0
    assert age_str2flt('9M') == 0.75
    """
    #
    if not age_str:
        raise ValueError("Undefined age string")
    if "岁" in age_str:
        y_tok = "岁"
    elif "年" in age_str:
        y_tok = "年"
    elif "Y" in age_str:
        y_tok = "Y"
    elif "y" in age_str:
        y_tok = "y"
    else:
        y_tok = None
    #
    if "月" in age_str:
        m_tok = "月"
    elif "M" in age_str:
        m_tok = "M"
    elif "m" in age_str:
        m_tok = "m"
    else:
        m_tok = None
    #
    y, m = 0, 0
    if y_tok:
        tokens = age_str.split(y_tok)
        assert len(tokens) <= 2
        y = int(tokens[0])
        age_str = tokens[1]
    if m_tok:
        tokens = age_str.split(m_tok)
        assert len(tokens) <= 2
        m = int(tokens[0])
    return y + m / 12.0


def select_13bones_from_ann(degrees_ann, decimal=0):
    """
    Input:
      degrees_ann: shape: 20 or Nx20
      decimal:     if not None, round the degree
    Output:
      13 rus degree in 'RUS keypoint order'
    """
    order_convert = [12, 9, 7, 4, 11, 8, 6, 3, 10, 5, 2, 1, 0]
    degrees_ann = np.array(degrees_ann)
    if len(degrees_ann.shape) == 1:
        degrees_13 = degrees_ann[order_convert]
    else:
        degrees_13 = degrees_ann[:, order_convert]
    if decimal is not None:
        degrees_13 = degrees_13.round(decimal)
    return degrees_13


def convert_2_standard_rus(degrees_kp, decimal=None):
    """
    internel keypoint model order -> standard annotation order
    """
    order_convert = [12, 11, 10, 7, 3, 9, 6, 2, 5, 1, 8, 4, 0]
    degrees_kp = np.array(degrees_kp)
    if len(degrees_kp.shape) == 1:
        degrees_13 = degrees_kp[order_convert]
    else:
        degrees_13 = degrees_kp[:, order_convert]
    if decimal is not None:
        degrees_13 = degrees_13.round(decimal)
    return degrees_13


def select_carpal_bones_from_ann(degrees_ann, decimal=0):
    """
    Input:
      degrees_ann: shape: 20 or Nx20
      decimal:     if not None, round the degree
    """
    degrees_ann = np.array(degrees_ann)
    if degrees_ann.ndim == 1:
        degrees_carpal = degrees_ann[13:]
    else:
        degrees_carpal = degrees_ann[:, 13:]
    if decimal is not None:
        degrees_carpal = degrees_carpal.round(decimal)
    return degrees_carpal


def growth_exception_handler_1d(degrees, verbose=False):
    degrees = np.array(degrees).copy()
    assert degrees.ndim == 1
    if degrees.shape[0] == 13:
        return growth_exception_handler_1d_13bones(degrees, verbose)
    elif degrees.shape[0] == 20:
        return growth_exception_handler_1d_20bones(degrees, verbose)
    else:
        assert 0, "degrees length invalid: %d" % degrees.shape[0]


def growth_exception_handler_1d_13bones(degree13, verbose=False):
    """
    13 keypoint format
    """
    exp_ref_map = {
        # five -> three
        0: 4,
        1: 5,
        2: 6,
        3: 7,
        # one -> three
        8: 4,
        9: 6
        # 10: 7
    }
    fpath = abspath(this_filename)
    if "DW_BoneAge" in fpath:
        from ba_utils.bone_age_errors import BoneAgeError
    else:
        from bone_age.ba_utils.bone_age_errors import BoneAgeError
    degree13 = np.array(degree13).copy()
    anchor_idx = [i for i in range(13) if i not in exp_ref_map]
    if -1 in degree13[anchor_idx]:
        raise BoneAgeError("growth exception not replaced")
    for idx, deg in enumerate(degree13):
        if deg == -1:
            ref_idx = exp_ref_map[idx]
            if verbose:
                print(
                    "\t发现生理性变异: %s => 参照 %s : 等级"
                    % (bone_names_13[idx], bone_names_13[ref_idx]),
                    degree13[ref_idx],
                )
            degree13[idx] = degree13[ref_idx]
    return list(degree13)


def growth_exception_handler_1d_20bones(degree20, verbose=False):
    """
    Standard RUS + Carpal Annotation Order
    """
    exp_ref_map = {
        # five -> three
        12: 11,
        9: 8,
        7: 6,
        4: 3,
        # one -> three
        10: 11,
        5: 6,
    }
    fpath = abspath(this_filename)
    if "DW_BoneAge" in fpath:
        from ba_utils.bone_age_errors import BoneAgeError
    else:
        from bone_age.ba_utils.bone_age_errors import BoneAgeError
    degree20 = np.array(degree20).copy()
    for idx, deg in enumerate(degree20):
        if deg == -1:
            if idx not in exp_ref_map:
                if verbose:
                    print("\t发现生理性变异: %s : 不可替换-跳过该标注" % (bone_names_20[idx]))
                raise BoneAgeError("growth exception could not be replaced")
            else:
                ref_idx = exp_ref_map[idx]
                if verbose:
                    print(
                        "\t发现生理性变异: %s => 参照 %s : 等级"
                        % (bone_names_20[idx], bone_names_20[ref_idx]),
                        degree20[ref_idx],
                    )
                degree20[idx] = degree20[ref_idx]
    return list(degree20)


def align_np2d(degree_NxB):
    degree_NxB = np.array(degree_NxB)
    if degree_NxB.ndim < 2:
        n_bone = degree_NxB.shape[0]
        degree_NxB = degree_NxB[np.newaxis, :]
    assert degree_NxB.ndim == 2
    return degree_NxB


def growth_exception_handler(degree_NxB, exception_tag="", verbose=False):
    fpath = abspath(this_filename)
    if "DW_BoneAge" in fpath:
        from ba_utils.bone_age_errors import BoneAgeError
    else:
        from bone_age.ba_utils.bone_age_errors import BoneAgeError
    """
    degree_NxB
        N >= 1:  number of doctors
        B == 13: RUS 13-keypoint format
        B == 20: standard annotation format
    """
    degree_NxB = align_np2d(degree_NxB)
    n_doc, n_bone = degree_NxB.shape
    assert n_bone in [13, 20]
    degree_valid_2d = []
    for i in range(n_doc):
        try:
            degree_valid = growth_exception_handler_1d(degree_NxB[i], verbose)
        except:
            if verbose:
                print("[发育异常] 跳过:%s" % exception_tag)
            continue
        for i, deg in enumerate(degree_valid):
            if deg == -1:
                assert 0, "growth exception epiphysis not well handled"
        degree_valid_2d.append(degree_valid)
    if len(degree_valid_2d) == 0:
        raise BoneAgeError(
            "no qualified degree label left after growth exception handler!"
        )
    return np.array(degree_valid_2d)


# --------------------------------------------------------------------------------
def get_surround_gray_val(img_origin):
    img = img_origin.copy()
    if len(img.shape) == 3:
        img = img[:, :, 0]

    h, w = img.shape
    h_width = int(round(h / 10.0))
    w_width = int(round(w / 10.0))
    m1 = np.mean(img[h_width : 2 * h_width])
    m2 = np.mean(img[-2 * h_width : -h_width])
    m3 = np.mean(img[:, w_width : 2 * w_width])
    m4 = np.mean(img[:, -2 * w_width : -w_width])
    m = (m1 + m2 + m3 + m4) / 4.0
    return m


def reflect_image(image_np8):
    reflected = False
    if get_surround_gray_val(image_np8) > 120:
        image_np8 = 255 - image_np8
        reflected = True
    return image_np8, reflected


def get_all_files_under(
    input_path,
    with_ext=None,
    skip_ext=["json", "txt", "DICOMDIR"],
    skip_prefix=[".DS_Store"],
):
    fpath_list = []
    input_path = convert2unicode(input_path)
    for dirpath, dirnames, filenames in os.walk(input_path, followlinks=True):
        for fname in filenames:
            skip_this_file = False
            for ext in skip_ext:
                if fname.endswith(ext):
                    skip_this_file = True
                    break
            for prefix in skip_prefix:
                if fname.startswith(prefix):
                    skip_this_file = True
                    break
            if skip_this_file:
                continue

            if with_ext is not None and len(with_ext) > 0:
                select_this_file = False
                for ext in with_ext:
                    if fname.endswith(ext):
                        select_this_file = True
                        break
                if not select_this_file:
                    continue
            fpath = join(dirpath, fname)
            if isfile(fpath):
                fpath_list.append(fpath)
    print("Total number of files: ", len(fpath_list))
    return fpath_list


def get_all_dcm_under(dcm_top_dirs):
    dcm_path_list = []
    for dcm_top_dir in dcm_top_dirs:
        dcm_path_list.extend(
            get_all_files_under(
                dcm_top_dir, with_ext=[".dcm", ".DCM", ".dicom", ".DICOM"]
            )
        )
    return dcm_path_list


def wc2mm(w, c):
    w, c = float(w), float(c)
    return c - w / 2.0, c + w / 2.0


def mm2wc(win_min, win_max):  #
    win_min, win_max = float(win_min), float(win_max)
    win_width = win_max - win_min
    win_center = (win_max + win_min) / 2.0
    return win_width, win_center


def get_continuous_index(index_list, weight=None, verbose=None):  #
    if weight is None:
        # by yinzihao
        # print(index_list)
        index_difference = index_list[1:] - index_list[:-1]
        continuous_index = index_list[np.where(index_difference == 1)]
        return np.min(continuous_index), np.max(continuous_index) + 1
    else:
        # by gongping
        island_list = []  # list of tuple (length, weight, start_idx, end_idx)
        start_idx = index_list[0]
        end_idx = index_list[0]
        for idx in index_list[1:]:
            if idx != end_idx + 1:
                # new island
                island_length = end_idx - start_idx + 1
                island_weight = np.sum(weight[start_idx : end_idx + 1])
                island_list.append((island_length, island_weight, start_idx, end_idx))
                # init for next island
                start_idx = idx
                end_idx = idx
            else:
                # extend current island
                end_idx += 1
        island_length = end_idx - start_idx + 1
        island_weight = np.sum(weight[start_idx : end_idx + 1])
        island_list.append((island_length, island_weight, start_idx, end_idx))
        # find the main land
        island_list = sorted(island_list, reverse=True)
        if verbose:
            for island in island_list:
                print("island: (length, weight, start, end)", island)
        main_land = island_list[0]
        return main_land[2], main_land[3]


def crop_bbox_from_img(img_np, win_bbox):  #
    x, y, w, h = map(int, win_bbox)
    if len(img_np.shape) == 3:
        crop_np = img_np[y : y + h, x : x + w, :]
    elif len(img_np.shape) == 2:
        crop_np = img_np[y : y + h, x : x + w]
    else:
        assert False, "Invalid image dimension"

    return crop_np


glb_default_cutoff_ratio = 0.05


def get_default_cutoff_ratio():  #
    global glb_default_cutoff_ratio
    return glb_default_cutoff_ratio


def set_default_cutoff_ratio(cutoff_ratio):
    global glb_default_cutoff_ratio
    glb_default_cutoff_ratio = cutoff_ratio


def calc_window_from_img(
    img_np,
    cutoff_ratio=None,
    min_win_energe=0.1,
    ext_win=0.05,
    win_mode="alg",
    win_bbox=None,
    out_mode="wc",
    num_bins=100,
    weight_sigma=None,
    weight_truncate=4,
    verbose=False,
):  #
    """
    Calculate window width and window center from histogram of numpy image

    'win_mode':  'alg': 100bin connectivity guess
                 'mm': min/max window
    'out_mode':  'wc': window width/center
                 'mm': min/max window

    if win_mode == 'alg'
      100bins
      cutoff_ratio: bin threshold 1/100 * 0.05:
      min_win_energe: if within window energe < min_win_energe, fallback to (min, max) window
      ext_window by 5% per side: ext_win=0.05
    """
    if cutoff_ratio is None:
        cutoff_ratio = get_default_cutoff_ratio()

    if win_bbox is not None:
        img_np = crop_bbox_from_img(img_np, win_bbox)

    ori_min, ori_max = np.min(img_np), np.max(img_np)
    if win_mode == "mm":
        return mm2wc(ori_min, ori_max)
    pixel_span = int(ori_max - ori_min)
    bin_num = num_bins
    if pixel_span < bin_num:
        bin_num = pixel_span
    cnt, bins = np.histogram(img_np, bins=bin_num)
    weight = cnt / float(np.sum(cnt))
    if weight_sigma:
        weight = gaussian_filter1d(
            weight, weight_sigma, mode="nearest", truncate=weight_truncate
        )
    intervals = np.array(list(zip(bins[:-1], bins[1:])))
    mean_weight = 1.0 / bin_num
    index = np.where(weight >= mean_weight * cutoff_ratio)[0]
    min_index, max_index = get_continuous_index(index, weight=weight)
    within_window_energe = weight[min_index : max_index + 1].sum()
    if within_window_energe < min_win_energe:
        vmin, vmax = ori_min, ori_max
    else:
        vmin = intervals[min_index][0]
        vmax = intervals[max_index][1]
        win_width = vmax - vmin
        if ext_win != 0 and verbose:
            print("ext window: %.1f, %.1f ->" % (vmin, vmax), end=" ")
        vmin = vmin - win_width * ext_win
        vmax = vmax + win_width * ext_win
        if ext_win != 0 and verbose:
            print("%.1f, %.1f" % (vmin, vmax))

    if out_mode == "wc":
        return mm2wc(vmin, vmax)
    elif out_mode == "mm":
        return vmin, vmax
    else:
        assert False


def extract_dcm_from_dirs(dicom_dir_list):
    dcm_path_list = []
    for dicom_dir in dicom_dir_list:
        path_list = glob.glob(os.path.join(dicom_dir, "*/*/*/*.dcm"))
        dcm_path_list += path_list
    print("Total Number of DICOMs within `dicom_dir_list`", len(dcm_path_list))
    return dcm_path_list


def apply_window(img_u16, win_wc):
    w, c = win_wc
    img_u16 = img_u16.copy()
    img_u16 = img_u16.astype(np.float)
    win_up = c + w / 2
    win_down = c - w / 2
    img_u16[img_u16 > win_up] = win_up
    img_u16[img_u16 < win_down] = win_down
    w = win_up - win_down
    img_u16 -= win_down
    img_8u = np.array(img_u16 * 255.0 / w, dtype="uint8")
    return img_8u


def extract_foreground_from_pure_background(img2d_np, verbose=False):
    assert img2d_np.ndim == 2, "expecting 2d image: HxW"
    pixel_max, pixel_min = img2d_np.max(), img2d_np.min()
    if verbose:
        print("pixel max: %d, pixel min: %d" % (pixel_max, pixel_min))
    assert pixel_max > pixel_min, "error.pure_graylevel_image"
    fg_idx = np.where((img2d_np != pixel_max) & (img2d_np != pixel_min))
    min_h, max_h = fg_idx[0].min(), fg_idx[0].max()
    min_w, max_w = fg_idx[1].min(), fg_idx[1].max()
    if verbose:
        print("Original shape: ", img2d_np.shape)
        print("Foreground height: ", min_h, max_h)
        print("Foreground  width: ", min_w, max_w)
    img_np_fg = img2d_np[min_h : max_h + 1, min_w : max_w + 1]
    return img_np_fg, min_h, max_h, min_w, max_w


def apply_window_to_dcm(dicom_path, extract_fg=True, win_mode="alg"):
    if sys.version_info < (3, 0):
        origin_image_itk = sitk.ReadImage(dicom_path.encode("utf-8"))
    else:
        origin_image_itk = sitk.ReadImage(dicom_path)
    origin_image_np16 = sitk.GetArrayFromImage(origin_image_itk)[0]
    dcm_info = get_dcm_info(origin_image_itk)
    dcm_info["dcm_height"] = origin_image_np16.shape[0]
    dcm_info["dcm_width"] = origin_image_np16.shape[1]
    if extract_fg:
        (
            origin_image_np16,
            min_h,
            max_h,
            min_w,
            max_w,
        ) = extract_foreground_from_pure_background(origin_image_np16)
        dcm_info["fg_height"] = max_h - min_h + 1
        dcm_info["fg_width"] = max_w - min_w + 1
        dcm_info["fg_top_wh"] = min_w, min_h
        dcm_info["fg_bot_wh"] = max_w, max_h
    win_width, win_center = calc_window_from_img(origin_image_np16, win_mode=win_mode)
    dcm_info["cal_win_width"] = win_width
    dcm_info["cal_win_center"] = win_center
    image_np8 = apply_window(origin_image_np16, (win_width, win_center))
    return origin_image_itk, origin_image_np16, image_np8, dcm_info


def apply_window_to_img(img, **kwargs):  #
    win_width, win_center = calc_window_from_img(img, out_mode="wc", **kwargs)
    img_np8 = apply_window(img, (win_width, win_center))
    return img_np8


def mm_is_inside_wc(min_max, win_wc):
    w_min, w_max = wc2mm(*win_wc)
    if (w_min <= min_max[0] <= w_max) and (w_min <= min_max[1] <= w_max):
        return True, win_wc
    else:
        new_min = min(w_min, min_max[0])
        new_max = max(w_max, min_max[1])
        new_width, new_center = mm2wc(new_min, new_max)
        return False, (new_width, new_center)


dcm_meta_keys = OrderedDict(
    [
        ("patient_age", "0010|1010"),
        ("birth_date", "0010|0030"),
        ("series_date", "0008|0021"),
        ("study_date", "0008|0020"),
        ("patient_name", "0010|0010"),
        ("sex", "0010|0040"),
        ("dcm_win_center", "0028|1050"),
        ("dcm_win_width", "0028|1051"),
        ("patient_id", "0010|0020"),
        ("study_id", "0020|000d"),
        ("series_id", "0020|000e"),
        ("sop_id", "0008|0018"),
        ("instance_number", "0020|0013"),
        ("institution_name", "0008|0080"),
        ("study_desc", "0008|1030"),
        ("series_desc", "0008|103e"),
        ("body_part", "0018|0015"),
        ("modality", "0008|0060"),
        ("manufacturer", "0008|0070"),
        ("model_name", "0008|1090"),
        ("KVP", "0018|0060"),
        ("Tube Current(mA)", "0018|1151"),
        ("Exposure Time(msec)", "0018|1150"),
        ("Exposure(mAs)", "0018|1152"),
        ("SourceImageDist(mm)", "0018|1110"),
        ("EntranceDose(dGy)", "0040|0302"),
        ("OrganDose(dGy)", "0040|0316"),
    ]
)


def get_dcm_info(image_sitk):
    image = image_sitk
    img_meta_data_keys = image.GetMetaDataKeys()
    info_dict = {}
    for k, v in dcm_meta_keys.items():
        dcm_val = None
        if v in img_meta_data_keys:
            if int(sys.version[0]) < 3:
                dcm_val = (image.GetMetaData(v.encode("utf-8"))).strip()
            else:
                dcm_val = (image.GetMetaData(v)).strip()
            if len(dcm_val) == 0:
                dcm_val = None
        if k in ["patient_name", "patient_id", "body_part"] and dcm_val is not None:
            decode_ok = False
            try:
                if int(sys.version[0]) < 3:
                    dcm_val = dcm_val.decode("utf-8")
                else:
                    dcm_val = dcm_val
                decode_ok = True
                # print('[utf-8] %s: %s' % (k, dcm_val))
            except UnicodeDecodeError as e:
                dcm_val = dcm_val.decode("gbk")
                decode_ok = True
                # print('[gbk] %s: %s' % (k, dcm_val))
            info_dict["%s_decode_ok" % k] = decode_ok
        info_dict[k] = dcm_val
    # for k, v in info_dict.items():
    #    print('\t%s: %s' % (k, v))
    info_dict["dcm_height"] = image.GetHeight()
    info_dict["dcm_width"] = image.GetWidth()
    return info_dict


def get_study_date(dcm_info):
    if "study_date" in dcm_info:
        return dcm_info["study_date"]
    elif "series_date" in dcm_info:
        return dcm_info["series_date"]
    else:
        return "Unknown"


def get_sub_dir_from_dcm_info(dcm_info, use_sop_id=True):
    id_list = [dcm_info["patient_id"], dcm_info["study_id"], dcm_info["series_id"]]
    if use_sop_id:
        if not id_list[0].startswith("ba_p1_"):
            id_list = id_list + [dcm_info["sop_id"]]
    sub_dir = "/".join(id_list)
    return sub_dir


def parse_dcm_list_by_sitk(input_dcm_path_list, use_sop_id=True, extract_fg=True):
    """
    Input:
        list of file paths
    Output:
        dcm_info_dict:    sub_dir -> {'patient_id': ..., 'modality': ...,}
        read_failed_list: list of read failed pathes
    """
    read_failed_list = []
    dcm_info_dict = {}
    total_file_num = len(input_dcm_path_list)
    for i, dcm_path in enumerate(input_dcm_path_list):
        print("[%4d/%4d] %s" % (i, total_file_num, dcm_path))
        try:
            (
                origin_image_itk,
                origin_image_np16,
                image_np8,
                dcm_info,
            ) = apply_window_to_dcm(dcm_path, extract_fg=extract_fg)
            sub_dir = get_sub_dir_from_dcm_info(dcm_info, use_sop_id=use_sop_id)
            dcm_info["path"] = dcm_path
            if sub_dir in dcm_info_dict:
                print("---- conflict with %s ----" % dcm_info_dict[sub_dir]["path"])
                print("")
            dcm_info_dict[sub_dir] = dcm_info
        except Exception as ex:
            import traceback

            traceback.print_exc()
            print("sitk.ReadImage failed: ", dcm_path)
            read_failed_list.append(dcm_path)
            continue
    print("-" * 80)
    print("Failed: %d/%d" % (len(read_failed_list), total_file_num))
    return dcm_info_dict, read_failed_list


def count_body_part(dcm_info_dict):
    body_part_dict = {}
    for sub_dir in sorted(dcm_info_dict.keys()):
        dcm_info = dcm_info_dict[sub_dir]
        body_part = dcm_info["body_part"]
        if body_part in body_part_dict:
            body_part_dict[body_part].append(sub_dir)
        else:
            body_part_dict[body_part] = [sub_dir]
    return body_part_dict


def show_body_part(part_dict):
    part_cnt_list = [
        (part_name, len(part_list)) for part_name, part_list in part_dict.items()
    ]
    part_cnt_list = sorted(part_cnt_list, key=lambda tup: tup[1])
    for name, cnt in part_cnt_list:
        try:
            print("Body Part %20s: %d" % (name, cnt))
        except UnicodeEncodeError:
            continue


def count_dcm_attr(attr_name, dcm_info_dict):
    attr_sub_dir_dict = {}
    for sub_dir in sorted(dcm_info_dict.keys()):
        dcm_info = dcm_info_dict[sub_dir]
        attr_val = dcm_info[attr_name]
        if attr_val in attr_sub_dir_dict:
            attr_sub_dir_dict[attr_val].append(sub_dir)
        else:
            attr_sub_dir_dict[attr_val] = [sub_dir]
    return attr_sub_dir_dict


def show_dcm_attr(attr_name, attr_sub_dir_dict):
    attr_cnt_list = [
        (attr_val, len(sub_dir_list))
        for attr_val, sub_dir_list in attr_sub_dir_dict.items()
    ]
    attr_cnt_list = sorted(attr_cnt_list, key=lambda tup: tup[1])
    for attr_val, cnt in attr_cnt_list:
        print("%s %20s: %d" % (attr_name, attr_val, cnt))


def parse_all_dcm_by_sitk(
    input_root,
    save_dcm_info=True,
    save_failed_list=True,
    max_limit=None,
    use_sop_id=True,
):
    """
    Input:
        input_root: top level directory containing dcms

    Output:
        dcm_info_dict:    sub_dir -> {'patient_id': ..., 'modality': ...,}
        read_failed_list: list of read failed pathes
    """
    input_dcm_path_list = get_all_files_under(
        input_root, skip_ext=["json", "txt", "DICOMDIR", "xlsx", "xls"]
    )
    if max_limit is not None:
        input_dcm_path_list = sorted(input_dcm_path_list)[:max_limit]
    dcm_info_dict, read_failed_list = parse_dcm_list_by_sitk(
        input_dcm_path_list, use_sop_id=use_sop_id
    )
    if save_dcm_info:
        save_fname = normpath(input_root) + "_dcm_info.pkl"
        save_pickle(dcm_info_dict, save_fname)
    if save_failed_list:
        save_fname = normpath(input_root) + "_failed.pkl"
        save_pickle(read_failed_list, save_fname)
    print("-" * 80)
    print("-", input_root)
    part_dict = count_body_part(dcm_info_dict)
    show_body_part(part_dict)
    return dcm_info_dict, read_failed_list


def dcm_transformer(
    dcm_info_dict,
    npz_savedir=None,
    png_savedir=None,
    dcm_savedir=None,
    dcm_info_filter=None,
    img_transformer=None,
    extract_fg=False,
    win_mode="mm",
):
    if not isinstance(dcm_info_dict, dict) and isfile(dcm_info_dict):
        dcm_info_dict = load_pickle(dcm_info_dict)
    total_dcm_num = len(dcm_info_dict)
    failed_sub_dir_list = []
    for i, sub_dir in enumerate(sorted(dcm_info_dict.keys())):
        dcm_info = dcm_info_dict[sub_dir]
        if dcm_info_filter is not None:
            if not dcm_info_filter(dcm_info):
                continue
        print("[%4d/%4d] %s" % (i, total_dcm_num, sub_dir))
        try:
            if npz_savedir is not None or png_savedir is not None:
                (
                    origin_image_itk,
                    origin_image_np16,
                    image_np8,
                    _dcm_info,
                ) = apply_window_to_dcm(
                    dcm_info["path"], extract_fg=extract_fg, win_mode=win_mode
                )
                if img_transformer is not None:
                    image_np8 = img_transformer(image_np8)
        except:
            failed_sub_dir_list.append(sub_dir)
            import traceback

            traceback.print_exc()
            print("FAILED")
            continue
        if npz_savedir is not None:
            npz_savepath = join(npz_savedir, sub_dir)
            if os.path.exists(npz_savepath) is False:
                os.makedirs(npz_savepath)
                np.savez(join(npz_savepath, "img.npz"), image_np8)
        # png
        if png_savedir is not None:
            png_savedir = os.path.normpath(png_savedir)
            png_series_path = join(png_savedir + "_" + win_mode, sub_dir)
            if os.path.exists(png_series_path) is False:
                os.makedirs(png_series_path)
            info_fname = join(png_series_path, "info.pkl")
            save_pickle(_dcm_info, info_fname)
            # write transformed png to path
            png_fname = join(png_series_path, "img.png")
            cv2.imwrite(png_fname, image_np8)
            # softlink transformed png
            png_link_dir = png_savedir + "_" + win_mode + "_view"
            if not os.path.exists(png_link_dir):
                os.makedirs(png_link_dir)
            png_link_fname = join(png_link_dir, sub_dir.replace("/", "|") + ".png")
            if not os.path.exists(png_link_fname):
                os.symlink(png_fname, png_link_fname)
        # dcm: softlink
        if dcm_savedir is not None:
            dcm_link_dir = join(dcm_savedir, sub_dir)
            if not exists(dcm_link_dir):
                os.makedirs(dcm_link_dir)
            dcm_link_fname = join(dcm_link_dir, "img.dcm")
            if os.path.islink(dcm_link_fname):
                os.remove(dcm_link_fname)
            os.symlink(dcm_info["path"], dcm_link_fname)
    return failed_sub_dir_list


def reorganize_all_dcm_under(
    dcm_input_root,
    npz_savedir=None,
    png_savedir=None,
    dcm_savedir=None,
    use_sop_id=False,
    dcm_info_filter=None,
    img_transformer=None,
    extract_fg=False,
    **kwargs
):
    dcm_info_dict, failed_read_list = parse_all_dcm_by_sitk(
        dcm_input_root, save_dcm_info=True, save_failed_list=True, use_sop_id=use_sop_id
    )
    if npz_savedir is not None or png_savedir is not None or dcm_savedir is not None:
        dcm_transformer(
            dcm_info_dict,
            npz_savedir,
            png_savedir,
            dcm_savedir,
            dcm_info_filter=dcm_info_filter,
            img_transformer=img_transformer,
            extract_fg=extract_fg,
            **kwargs
        )


def check_hand_for_dataset(data_top, sub_dir_list, verbose=False):
    # data_root = '/data1/bone_age/data'
    no_keypoint_list = []
    is_hand_list, not_hand_list = [], []
    for idx, sub_dir in enumerate(sub_dir_list):
        # print('[%2d] %s' % (idx, sub_dir_list[idx]))
        key_info_fname = join(data_top, "label", sub_dir, "label.pickle")
        if not os.path.isfile(key_info_fname):
            print("[WARN] no label file: %s" % key_info_fname)
            no_keypoint_list.append(sub_dir)
        else:
            key_info = load_pickle(key_info_fname)
            if key_info["is_hand"]:
                is_hand_list.append(sub_dir)
            else:
                not_hand_list.append(sub_dir)
    if verbose:
        print(
            "check_hand_for_dataset: %s: sub_dir %d: is_hand: %d, not_hand: %d, no_keypoint_list: %d"
            % (
                data_top,
                len(sub_dir_list),
                len(is_hand_list),
                len(not_hand_list),
                len(no_keypoint_list),
            )
        )
    return is_hand_list, not_hand_list, no_keypoint_list


"""
p1_sub_dir_all = ba_utils.sub_dir_util.extract_sub_dir('/data1/bone_age/data/p1/dcm', sub_dir_level=3)
p1_sub_dir_map = {}
for sub_dir in p1_sub_dir_all:
    pid, stid, seid = sub_dir.split('/')
    p1_sub_dir_map['%s/0/0' % pid] = sub_dir
len(p1_sub_dir_map)
with open('/data1/bone_age/data/p1/sub_dir/p1_sub_dir_map.json', 'w') as f:
    json.dump(p1_sub_dir_map, f)
"""


def map_sub_dir_for_p1(sub_dir_list):
    p1_sub_dir_map = json.load(
        open("/data1/bone_age/data/p1/sub_dir/p1_sub_dir_map.json")
    )
    real_sub_dir_list = [p1_sub_dir_map[sub_dir] for sub_dir in sub_dir_list]

    return real_sub_dir_list


def get_correction_for_1st_finger(kp15, hand_side, verbose=False):
    vec_1st = kp15[10] - kp15[11]
    unit_vec_1st = vec_1st / np.sqrt(vec_1st[0] ** 2 + vec_1st[1] ** 2)
    if verbose:
        print("unit vector of 1st finger: %s" % (unit_vec_1st))
    ux, uy = unit_vec_1st
    if hand_side == "right":
        vx, vy = -uy, ux
    else:
        vx, vy = uy, -ux
    v_vec_1st = np.array([vx, vy])

    delta_kp10 = v_vec_1st * 15.0 + unit_vec_1st * -10.0
    if verbose:
        print("vertical vector of 1st finger: %f %f" % (vx, vy))
        print("detlta %s" % (delta_kp10))
    return delta_kp10
