#
# Copyright (C) 2019 Databricks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import pandas as pd
import numpy as np
import re

from databricks import koalas as ks
from databricks.koalas.testing.utils import ReusedSQLTestCase, SQLTestUtils


class SeriesStringTest(ReusedSQLTestCase, SQLTestUtils):
    @property
    def pser(self):
        return pd.Series(
            [
                "apples",
                "Bananas",
                "carrots",
                "1",
                "100",
                "",
                "\nleading-whitespace",
                "trailing-Whitespace    \t",
                None,
                np.NaN,
            ]
        )

    def check_func(self, func, almost=False):
        self.check_func_on_series(func, self.pser, almost=almost)

    def check_func_on_series(self, func, pser, almost=False):
        self.assert_eq(func(ks.from_pandas(pser)), func(pser), almost=almost)

    def test_string_add_str_num(self):
        pdf = pd.DataFrame(dict(col1=["a"], col2=[1]))
        kdf = ks.from_pandas(pdf)
        with self.assertRaises(TypeError):
            kdf["col1"] + kdf["col2"]

    def test_string_add_assign(self):
        pdf = pd.DataFrame(dict(col1=["a", "b", "c"], col2=["1", "2", "3"]))
        kdf = ks.from_pandas(pdf)
        kdf["col1"] += kdf["col2"]
        pdf["col1"] += pdf["col2"]
        self.assert_eq(kdf["col1"], pdf["col1"])

    def test_string_add_str_str(self):
        pdf = pd.DataFrame(dict(col1=["a", "b", "c"], col2=["1", "2", "3"]))
        kdf = ks.from_pandas(pdf)

        # TODO: Fix the Series names
        self.assert_eq(kdf["col1"] + kdf["col2"], (pdf["col1"] + pdf["col2"]).rename("col1"))
        self.assert_eq(kdf["col2"] + kdf["col1"], (pdf["col2"] + pdf["col1"]).rename("col2"))

    def test_string_add_str_lit(self):
        pdf = pd.DataFrame(dict(col1=["a", "b", "c"]))
        kdf = ks.from_pandas(pdf)
        self.assert_eq(kdf["col1"] + "_lit", pdf["col1"] + "_lit")
        self.assert_eq("_lit" + kdf["col1"], "_lit" + pdf["col1"])

    def test_string_capitalize(self):
        self.check_func(lambda x: x.str.capitalize())

    def test_string_title(self):
        self.check_func(lambda x: x.str.title())

    def test_string_lower(self):
        self.check_func(lambda x: x.str.lower())

    def test_string_upper(self):
        self.check_func(lambda x: x.str.upper())

    def test_string_swapcase(self):
        self.check_func(lambda x: x.str.swapcase())

    def test_string_startswith(self):
        pattern = "car"
        self.check_func(lambda x: x.str.startswith(pattern))
        self.check_func(lambda x: x.str.startswith(pattern, na=False))

    def test_string_endswith(self):
        pattern = "s"
        self.check_func(lambda x: x.str.endswith(pattern))
        self.check_func(lambda x: x.str.endswith(pattern, na=False))

    def test_string_strip(self):
        self.check_func(lambda x: x.str.strip())
        self.check_func(lambda x: x.str.strip("es\t"))
        self.check_func(lambda x: x.str.strip("1"))

    def test_string_lstrip(self):
        self.check_func(lambda x: x.str.lstrip())
        self.check_func(lambda x: x.str.lstrip("\n1le"))
        self.check_func(lambda x: x.str.lstrip("s"))

    def test_string_rstrip(self):
        self.check_func(lambda x: x.str.rstrip())
        self.check_func(lambda x: x.str.rstrip("\t ec"))
        self.check_func(lambda x: x.str.rstrip("0"))

    def test_string_get(self):
        self.check_func(lambda x: x.str.get(6))
        self.check_func(lambda x: x.str.get(-1))

    def test_string_isalnum(self):
        self.check_func(lambda x: x.str.isalnum())

    def test_string_isalpha(self):
        self.check_func(lambda x: x.str.isalpha())

    def test_string_isdigit(self):
        self.check_func(lambda x: x.str.isdigit())

    def test_string_isspace(self):
        self.check_func(lambda x: x.str.isspace())

    def test_string_islower(self):
        self.check_func(lambda x: x.str.islower())

    def test_string_isupper(self):
        self.check_func(lambda x: x.str.isupper())

    def test_string_istitle(self):
        self.check_func(lambda x: x.str.istitle())

    def test_string_isnumeric(self):
        self.check_func(lambda x: x.str.isnumeric())

    def test_string_isdecimal(self):
        self.check_func(lambda x: x.str.isdecimal())

    def test_string_cat(self):
        kser = ks.from_pandas(self.pser)
        with self.assertRaises(NotImplementedError):
            kser.str.cat()

    def test_string_center(self):
        self.check_func(lambda x: x.str.center(0))
        self.check_func(lambda x: x.str.center(10))
        self.check_func(lambda x: x.str.center(10, "x"))

    def test_string_contains(self):
        self.check_func(lambda x: x.str.contains("le", regex=False))
        self.check_func(lambda x: x.str.contains("White", case=True, regex=False))
        self.check_func(lambda x: x.str.contains("apples|carrots", regex=True))
        self.check_func(lambda x: x.str.contains("BANANAS", flags=re.IGNORECASE, na=False))

    def test_string_count(self):
        self.check_func(lambda x: x.str.count("wh|Wh"))
        self.check_func(lambda x: x.str.count("WH", flags=re.IGNORECASE))

    def test_string_decode(self):
        kser = ks.from_pandas(self.pser)
        with self.assertRaises(NotImplementedError):
            kser.str.decode("utf-8")

    def test_string_encode(self):
        kser = ks.from_pandas(self.pser)
        with self.assertRaises(NotImplementedError):
            kser.str.encode("utf-8")

    def test_string_extract(self):
        kser = ks.from_pandas(self.pser)
        with self.assertRaises(NotImplementedError):
            kser.str.extract("pat")

    def test_string_extractall(self):
        kser = ks.from_pandas(self.pser)
        with self.assertRaises(NotImplementedError):
            kser.str.extractall("pat")

    def test_string_find(self):
        self.check_func(lambda x: x.str.find("a"))
        self.check_func(lambda x: x.str.find("a", start=3))
        self.check_func(lambda x: x.str.find("a", start=0, end=1))

    def test_string_findall(self):
        self.check_func_on_series(lambda x: x.str.findall("es|as").apply(str), self.pser[:-1])
        self.check_func_on_series(
            lambda x: x.str.findall("wh.*", flags=re.IGNORECASE).apply(str), self.pser[:-1]
        )

    def test_string_index(self):
        pser = pd.Series(["tea", "eat"])
        self.check_func_on_series(lambda x: x.str.index("ea"), pser)
        with self.assertRaises(Exception):
            self.check_func_on_series(lambda x: x.str.index("ea", start=0, end=2), pser)
        with self.assertRaises(Exception):
            self.check_func(lambda x: x.str.index("not-found"))

    def test_string_join(self):
        pser = pd.Series([["a", "b", "c"], ["xx", "yy", "zz"]])
        self.check_func_on_series(lambda x: x.str.join("-"), pser)
        self.check_func(lambda x: x.str.join("-"))

    def test_string_len(self):
        self.check_func(lambda x: x.str.len())
        pser = pd.Series([["a", "b", "c"], ["xx"], []])
        self.check_func_on_series(lambda x: x.str.len(), pser)

    def test_string_ljust(self):
        self.check_func(lambda x: x.str.ljust(0))
        self.check_func(lambda x: x.str.ljust(10))
        self.check_func(lambda x: x.str.ljust(30, "x"))

    def test_string_match(self):
        self.check_func(lambda x: x.str.match("in"))
        self.check_func(lambda x: x.str.match("apples|carrots", na=False))
        self.check_func(lambda x: x.str.match("White", case=True))
        self.check_func(lambda x: x.str.match("BANANAS", flags=re.IGNORECASE))

    def test_string_normalize(self):
        self.check_func(lambda x: x.str.normalize("NFC"))
        self.check_func(lambda x: x.str.normalize("NFKD"))

    def test_string_pad(self):
        self.check_func(lambda x: x.str.pad(10))
        self.check_func(lambda x: x.str.pad(10, side="both"))
        self.check_func(lambda x: x.str.pad(10, side="right", fillchar="-"))

    def test_string_partition(self):
        with self.assertRaises(NotImplementedError):
            self.check_func(lambda x: x.str.partition())

    def test_string_repeat(self):
        self.check_func(lambda x: x.str.repeat(repeats=3))
        with self.assertRaises(ValueError):
            self.check_func(lambda x: x.str.repeat(repeats=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

    def test_string_replace(self):
        self.check_func(lambda x: x.str.replace("a.", "xx", regex=True))
        self.check_func(lambda x: x.str.replace("a.", "xx", regex=False))
        self.check_func(lambda x: x.str.replace("ing", "0", flags=re.IGNORECASE))
        # reverse every lowercase word
        repl = lambda m: m.group(0)[::-1]
        self.check_func(lambda x: x.str.replace(r"[a-z]+", repl))
        # compiled regex with flags
        regex_pat = re.compile(r"WHITESPACE", flags=re.IGNORECASE)
        self.check_func(lambda x: x.str.replace(regex_pat, "---"))

    def test_string_rfind(self):
        self.check_func(lambda x: x.str.rfind("a"))
        self.check_func(lambda x: x.str.rfind("a", start=3))
        self.check_func(lambda x: x.str.rfind("a", start=0, end=1))

    def test_string_rindex(self):
        pser = pd.Series(["teatea", "eateat"])
        self.check_func_on_series(lambda x: x.str.rindex("ea"), pser)
        with self.assertRaises(Exception):
            self.check_func_on_series(lambda x: x.str.rindex("ea", start=0, end=2), pser)
        with self.assertRaises(Exception):
            self.check_func(lambda x: x.str.rindex("not-found"))

    def test_string_rjust(self):
        self.check_func(lambda x: x.str.rjust(0))
        self.check_func(lambda x: x.str.rjust(10))
        self.check_func(lambda x: x.str.rjust(30, "x"))

    def test_string_rpartition(self):
        with self.assertRaises(NotImplementedError):
            self.check_func(lambda x: x.str.rpartition())

    def test_string_slice(self):
        self.check_func(lambda x: x.str.slice(start=1))
        self.check_func(lambda x: x.str.slice(stop=3))
        self.check_func(lambda x: x.str.slice(step=2))
        self.check_func(lambda x: x.str.slice(start=0, stop=5, step=3))

    def test_string_slice_replace(self):
        self.check_func(lambda x: x.str.slice_replace(1, repl="X"))
        self.check_func(lambda x: x.str.slice_replace(stop=2, repl="X"))
        self.check_func(lambda x: x.str.slice_replace(start=1, stop=3, repl="X"))

    def test_string_split(self):
        self.check_func_on_series(lambda x: x.str.split().apply(str), self.pser[:-1])
        self.check_func_on_series(lambda x: x.str.split(r"p*").apply(str), self.pser[:-1])
        pser = pd.Series(["This is a sentence.", "This-is-a-long-word."])
        self.check_func_on_series(lambda x: x.str.split(n=2).apply(str), pser)
        self.check_func_on_series(lambda x: x.str.split(pat="-", n=2).apply(str), pser)
        self.check_func_on_series(lambda x: x.str.split(n=2, expand=True), pser, almost=True)
        with self.assertRaises(NotImplementedError):
            self.check_func(lambda x: x.str.split(expand=True))

    def test_string_rsplit(self):
        self.check_func_on_series(lambda x: x.str.rsplit().apply(str), self.pser[:-1])
        self.check_func_on_series(lambda x: x.str.rsplit(r"p*").apply(str), self.pser[:-1])
        pser = pd.Series(["This is a sentence.", "This-is-a-long-word."])
        self.check_func_on_series(lambda x: x.str.rsplit(n=2).apply(str), pser)
        self.check_func_on_series(lambda x: x.str.rsplit(pat="-", n=2).apply(str), pser)
        self.check_func_on_series(lambda x: x.str.rsplit(n=2, expand=True), pser, almost=True)
        with self.assertRaises(NotImplementedError):
            self.check_func(lambda x: x.str.rsplit(expand=True))

    def test_string_translate(self):
        m = str.maketrans({"a": "X", "e": "Y", "i": None})
        self.check_func(lambda x: x.str.translate(m))

    def test_string_wrap(self):
        self.check_func(lambda x: x.str.wrap(5))
        self.check_func(lambda x: x.str.wrap(5, expand_tabs=False))
        self.check_func(lambda x: x.str.wrap(5, replace_whitespace=False))
        self.check_func(lambda x: x.str.wrap(5, drop_whitespace=False))
        self.check_func(lambda x: x.str.wrap(5, break_long_words=False))
        self.check_func(lambda x: x.str.wrap(5, break_on_hyphens=False))

    def test_string_zfill(self):
        self.check_func(lambda x: x.str.zfill(10))

    def test_string_get_dummies(self):
        with self.assertRaises(NotImplementedError):
            self.check_func(lambda x: x.str.get_dummies())
