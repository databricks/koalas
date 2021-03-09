#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Contains useful utils to generate reStructuredText files for Sphinx.
"""

import shutil
import sys
from distutils.version import LooseVersion
from urllib.error import HTTPError
from urllib.request import urlopen, Request
import json
import os
import re
import platform

import pypandoc
from pypandoc import pandoc_download

import databricks.koalas as ks

# OAuth key used for issuing requests against the GitHub API. If this is not defined, then requests
# will be unauthenticated. You should only need to configure this if you find yourself regularly
# exceeding your IP's unauthenticated request rate limit. You can create an OAuth key at
# https://github.com/settings/tokens. This script only requires the "public_repo" scope.
GITHUB_OAUTH_KEY = os.environ.get("GITHUB_OAUTH_KEY")


def get_json(url):
    """
    >>> get_json("https://api.github.com/repos/data")
    Traceback (most recent call last):
       ...
    urllib.error.HTTPError: HTTP Error 404: Not Found
    >>> isinstance(get_json("https://api.github.com/repos/databricks/koalas/releases"), list)
    True
    """
    try:
        request = Request(url)
        if GITHUB_OAUTH_KEY:
            request.add_header('Authorization', 'token %s' % GITHUB_OAUTH_KEY)

        response = urlopen(request).read()
        if isinstance(response, bytes):
            response = response.decode('utf-8')
        return json.loads(response)
    except HTTPError as e:
        if "X-RateLimit-Remaining" in e.headers and e.headers["X-RateLimit-Remaining"] == '0':
            print("Exceeded the GitHub API rate limit; see the instructions in " +
                  "dev/gendoc.py to configure an OAuth token for making authenticated " +
                  "GitHub requests.", sys.stderr)
        else:
            print("Unable to fetch URL, exiting: %s" % url, sys.stderr)
        raise


def list_releases_to_document(cur_version):
    """
    >>> list_releases_to_document("0.1.0")[0][0]
    'Version 0.1.0'
    """
    tag_url = "https://api.github.com/repos/databricks/koalas/releases"
    cur_version = "v" + cur_version
    releases = [(
        release['name'], release['tag_name'], release['body']
    ) for release in retry(get_json, url=tag_url) if not release['draft']]
    filtered = filter(
        lambda release: LooseVersion(release[1]) <= LooseVersion(cur_version), releases)
    return sorted(filtered, reverse=True, key=lambda release: LooseVersion(release[1]))


def gen_release_notes(path):
    """
    Generate reStructuredText files for "Release Notes". It generates 'index.rst' file and
    each rst file for each version's release note under 'whatsnew' directory.
    The contents are from Github release notes.
    """
    whatsnew_dir = "%s/whatsnew" % path
    shutil.rmtree(whatsnew_dir, ignore_errors=True)
    os.mkdir(whatsnew_dir)

    with open("%s/index.rst" % whatsnew_dir, "a", encoding="utf-8") as index_file:
        title = "Release Notes"

        index_file.write("=" * len(title))
        index_file.write("\n")
        index_file.write(title)
        index_file.write("\n")
        index_file.write("=" * len(title))
        index_file.write("\n")
        index_file.write("\n")
        index_file.write(".. toctree::")
        index_file.write("   :maxdepth: 1")
        index_file.write("\n")
        index_file.write("\n")

        for name, tag_name, body in list_releases_to_document(ks.__version__):
            release_doc = pypandoc.convert_text(body, "rst", format="md")

            # Make PR reference link pretty.
            # Replace ", #..." to ", `...<https://github.com/databricks/koalas/pull/...>`_"
            release_doc = re.sub(
                r', #(\d+)',
                r', `#\1 <https://github.com/databricks/koalas/pull/\1>`_', release_doc)
            # Replace "(#..." to "(`...<https://github.com/databricks/koalas/pull/...>`_"
            release_doc = re.sub(
                r'\(#(\d+)',
                r'(`#\1 <https://github.com/databricks/koalas/pull/\1>`_', release_doc)

            index_file.write("   " + tag_name)
            index_file.write("\n")
            index_file.write("\n")

            with open("%s/%s.rst" % (whatsnew_dir, tag_name), "a", encoding="utf-8") \
                    as release_file:
                release_file.write("=" * len(name))
                release_file.write("\n")
                release_file.write(name)
                release_file.write("\n")
                release_file.write("=" * len(name))
                release_file.write("\n")
                release_file.write("\n")
                release_file.write(release_doc)
                release_file.write("\n")
                release_file.write("\n")


def retry(f, num=3, args=(), **kwargs):
    """
    Retry the given function `f` with the given arguments `num` times
    """

    try:
        return f(*args, **kwargs)
    except Exception as exception:
        for i in range(num):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                exception = e
                print("Retrying '%s' (%s/%s) ..." % (f.__name__, i + 1, num))
        else:
            raise exception


def is_exe(path):
    """
    Check if a given path is an executable file.
    From: http://stackoverflow.com/a/377028
    """

    return os.path.isfile(path) and os.access(path, os.X_OK)


def which(program):
    """
    Find and return the given program by its absolute path or 'None' if the program cannot be found.
    From: http://stackoverflow.com/a/377028
    """

    fpath = os.path.split(program)[0]

    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ.get("PATH").split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None


def download_pandoc_if_needed(path):
    """
    Download pandoc that is used by pypandoc.
    """
    if which("pandoc"):
        print("* Pandoc executable was found in your path, skipping installing pandoc...")
        return

    pandoc_urls, _ = retry(pandoc_download._get_pandoc_urls, version="latest")
    pf = sys.platform
    if pf.startswith("linux"):
        pf = "linux"
        if platform.architecture()[0] != "64bit":
            raise RuntimeError("Linux pandoc is only compiled for 64bit.")

    if pf not in pandoc_urls:
        raise RuntimeError("Can't handle your platform (only Linux, Mac OS X, Windows).")

    filename = pandoc_urls[pf].split("/")[-1]
    os.environ["PATH"] = "%s:%s" % (path, os.environ["PATH"])

    # Seems like it always downloads to the current working directory, and then unpack to
    # the given path. Let's download to the given path as well.
    prev = os.getcwd()
    os.chdir(path)
    try:
        if not os.path.isfile(filename) or not os.path.isfile("pandoc"):
            def download_pandoc():
                try:
                    return pandoc_download.download_pandoc(targetfolder=path, version="1.19.1")
                except Exception as e:
                    if os.path.isfile(filename):
                        os.remove(filename)
                    raise e

            retry(download_pandoc)
    finally:
        os.chdir(prev)
