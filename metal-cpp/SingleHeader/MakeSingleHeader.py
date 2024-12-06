#!/usr/bin/python3

# --------------------------------------------------------------------------------------------------------------------------------------------------------------
#
# SingleHeader/MakeSingleHeader.py
#
# Copyright 2020-2021 Apple Inc.
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
# --------------------------------------------------------------------------------------------------------------------------------------------------------------

import argparse
import datetime
import logging
import os
import re
import subprocess
import sys

# --------------------------------------------------------------------------------------------------------------------------------------------------------------


class HeaderPrefix(object):
    __template = (
        "//\n"
        "// {file}\n"
        "//\n"
        "// {meta_data}\n"
        "//\n"
        "// Copyright 2020-2021 Apple Inc.\n"
        "//\n"
        '// Licensed under the Apache License, Version 2.0 (the "License");\n'
        "// you may not use this file except in compliance with the License.\n"
        "// You may obtain a copy of the License at\n"
        "//\n"
        "//     http://www.apache.org/licenses/LICENSE-2.0\n"
        "//\n"
        "// Unless required by applicable law or agreed to in writing, software\n"
        '// distributed under the License is distributed on an "AS IS" BASIS,\n'
        "// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
        "// See the License for the specific language governing permissions and\n"
        "// limitations under the License.\n"
        "//\n"
        "\n"
    )

    __template_commit = "Autogenerated from commit {commit}."
    __template_date = "Autogenerated on %B %d, %Y."

    def __init__(self, file):
        self.__file = file

    def __str__(self):
        return self.__template.format(
            file=self.__file, meta_data=self.__meta_data_string()
        )

    def __get_commit_hash(self):
        git_commit_hash = None

        try:
            git_dir = os.path.dirname(os.path.realpath(__file__))
            proc = subprocess.Popen(
                ["git", "rev-parse", "HEAD"],
                cwd=git_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            git_commit_hash = proc.stdout.read().decode("utf-8", "replace").strip()
        except:
            logging.error("Failed to determine git commit hash!")
            pass

        return git_commit_hash

    def __get_commit_string(self):
        meta_data = None
        git_commit_hash = self.__get_commit_hash()

        if git_commit_hash:
            meta_data = self.__template_commit.format(commit=git_commit_hash)

        return meta_data

    def __get_date_string(self):
        today = datetime.date.today()

        return today.strftime(self.__template_date)

    def __meta_data_string(self):
        meta_data = self.__get_commit_string()

        if not meta_data:
            meta_data = self.__get_date_string()

        return meta_data


# --------------------------------------------------------------------------------------------------------------------------------------------------------------


class SingleHeader(object):
    __pragma_once = "#pragma once\n\n"

    def __init__(self):
        self.__header_paths = list()

    def __str__(self):
        return self.process()

    def append(self, header_path):
        self.__header_paths.append(header_path)

    def process(self):
        out_header = self.__pragma_once

        self.__included_headers = set()
        self.__base_path = list()

        for header_path in self.__header_paths:
            out_header += self.__process_header(header_path)

        return self.__strip_empty_lines(out_header)

    def __read_header(self, path):
        path = os.path.realpath(path)

        try:
            f = open(path, "r")
        except:
            raise RuntimeError('Failed to open file "' + path + '" for read!')

        return f.read()

    def __strip_pragma_once(self, header):
        return re.sub("\\s*#pragma once\s*\\/\\/-*\\n", "", header)

    def __strip_comments(self, header):
        return re.sub("^//.*\\n", "", header, flags=re.MULTILINE)

    def __strip_empty_lines(self, header):
        return re.sub("\\n\\n+", "\\n\\n", header, flags=re.MULTILINE)

    def __substitute_include_directive(self, match):
        header_path = match.group("HEADER_PATH")

        logging.info('\tSubstituting "' + header_path + '"...')

        return self.__process_header(os.path.join(self.__base_path[-1], header_path))

    def __process_include_directives(self, header):
        return re.sub(
            '^\\s*#include\\s\\"(?P<HEADER_PATH>\\S*)\\"',
            self.__substitute_include_directive,
            header,
            flags=re.MULTILINE,
        )

    def __process_foundation_directives(self, header):
        if header.find("#include <Foundation/Foundation.hpp>") != -1:
            logging.info("\tSubstituting <Foundation/Foundation.hpp>...")
            return header.replace(
                "#include <Foundation/Foundation.hpp>",
                self.__process_header(
                    os.path.join(self.__base_path[-1], "../Foundation/Foundation.hpp")
                ),
            )
        return header

    def __process_header(self, header_path):
        out_header = ""

        header_path = os.path.realpath(header_path)

        if not header_path in self.__included_headers:
            logging.info('Processing "' + header_path + '"...')

            self.__base_path.append(os.path.dirname(header_path))
            self.__included_headers.add(header_path)

            out_header = self.__read_header(header_path)
            out_header = self.__strip_pragma_once(out_header)
            out_header = self.__strip_comments(out_header)
            out_header = self.__process_include_directives(out_header)
            out_header = self.__process_foundation_directives(out_header)

            self.__base_path.pop()
        else:
            logging.info('\tSkipping "' + header_path + '"...')

        return out_header


# --------------------------------------------------------------------------------------------------------------------------------------------------------------


def create_argument_parser():
    parser = argparse.ArgumentParser()
    base_path = os.path.dirname(os.path.realpath(__file__))
    output_path = os.path.join(base_path, "Metal.hpp")

    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        metavar="PATH",
        default=output_path,
        help="Output path for the single header file.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show verbose output."
    )
    parser.add_argument(
        dest="header_paths", metavar="HEADER_FILE", nargs="+", help="Input header file."
    )

    return parser


# --------------------------------------------------------------------------------------------------------------------------------------------------------------


def parse_arguments():
    parser = create_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.ERROR)

    return args


# --------------------------------------------------------------------------------------------------------------------------------------------------------------


def make_header(args):
    prefix = HeaderPrefix(os.path.basename(args.output_path))
    header = SingleHeader()

    for header_path in args.header_paths:
        header.append(header_path)

    return str(prefix) + str(header)


# --------------------------------------------------------------------------------------------------------------------------------------------------------------


def make_dir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except os.error:
        pass
    except:
        raise


# --------------------------------------------------------------------------------------------------------------------------------------------------------------


def write_header(args, content):
    path = os.path.realpath(args.output_path)

    logging.info('Writing "' + path + '"...')

    make_dir(os.path.dirname(path))

    try:
        f = open(path, "w")
    except:
        raise RuntimeError('Failed to open file "' + path + '" for write!')

    f.write(content)


# --------------------------------------------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    result = -1

    try:
        if sys.getdefaultencoding().lower() == "ascii":
            reload(sys)
            sys.setdefaultencoding("utf-8")

        args = parse_arguments()
        header = make_header(args)

        write_header(args, header)

        result = 0

    except (KeyboardInterrupt, SystemExit):
        pass
    except:
        raise

    sys.exit(result)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------
