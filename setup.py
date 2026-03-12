"""Python packaging file for ODML LiteRT LM library."""

import atexit
import contextlib
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

import setuptools
import setuptools.command.build_py
import setuptools.command.develop
import setuptools.command.egg_info
import setuptools.command.sdist


_build_py = setuptools.command.build_py.build_py
_develop = setuptools.command.develop.develop
_egg_info = setuptools.command.egg_info.egg_info
_sdist = setuptools.command.sdist.sdist


def compile_protos() -> None:
  """Compiles .proto files using grpc_tools.protoc."""
  proto_dir = "runtime/proto"
  proto_files = glob.glob(os.path.join(proto_dir, "*.proto"))
  if not proto_files:
    return
  os.makedirs("schema/py/runtime/proto", exist_ok=True)
  with open("schema/py/runtime/__init__.py", "a"):
    pass
  with open("schema/py/runtime/proto/__init__.py", "a"):
    pass

  for proto_file in proto_files:
    print(f"Compiling {proto_file}...")
    subprocess.check_call([
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        "-I.",
        "--python_out=schema/py",
        proto_file,
    ])

  # Modify the import paths in the generated files to match the package
  # structure. This makes the generated code importable as part of the
  # litert_lm package.
  pb2_files = glob.glob("schema/py/runtime/proto/*_pb2.py")
  for pb2_file in pb2_files:
    with open(pb2_file, "r") as f:
      content = f.read().replace(
          "from runtime.proto import", "from litert_lm.runtime.proto import"
      ).replace("runtime.proto.", "litert_lm.runtime.proto.")
    with open(pb2_file, "w") as f:
      f.write(content)


def _compile_with_flatc(executable_flatc: str) -> None:
  """Compiles .fbs files using the provided flatc executable."""
  fbs_dir = "schema/core"
  fbs_files = glob.glob(os.path.join(fbs_dir, "*.fbs"))
  if not fbs_files:
    return
  os.makedirs("schema/py/schema/core", exist_ok=True)
  with open("schema/py/schema/__init__.py", "a"):
    pass
  with open("schema/py/schema/core/__init__.py", "a"):
    pass

  for fbs_file in fbs_files:
    print(f"Compiling {fbs_file}...")
    subprocess.check_call([
        executable_flatc,
        "--python",
        "-o",
        "schema/py/schema/core",
        fbs_file,
    ])
  generated_dir = "schema/py/schema/core/litert/lm/schema"
  if os.path.exists(generated_dir):
    generated_modules = (
        f[:-3]
        for f in os.listdir(generated_dir)
        if f.endswith(".py") and f != "__init__.py"
    )
    with open(
        "schema/py/schema/core/litertlm_header_schema_py_generated.py", "w"
    ) as outfile:
      for mod in generated_modules:
        outfile.write(f"from litert.lm.schema.{mod} import *\n")


@contextlib.contextmanager
def _ensure_flatc_executable():
  """Finds or downloads the flatc compiler, yielding its path."""
  try_paths = ["./flatc", "../../flatc"]
  for path in try_paths:
    if os.path.exists(path):
      yield path
      return

  print("flatc not found. Downloading flatc...")
  with tempfile.TemporaryDirectory() as temp_dir:
    zip_path = os.path.join(temp_dir, "flatc.zip")
    url = "https://github.com/google/flatbuffers/releases/download/v23.5.26/Linux.flatc.binary.clang++-15.zip"
    try:
      urllib.request.urlretrieve(url, zip_path)
    except urllib.error.HTTPError:
      url = "https://github.com/google/flatbuffers/releases/download/v24.3.25/Linux.flatc.binary.clang++-15.zip"
      urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
      zip_ref.extractall(temp_dir)
    downloaded_flatc = os.path.join(temp_dir, "flatc")
    os.chmod(downloaded_flatc, 0o755)
    yield downloaded_flatc


def compile_flatbuffers() -> None:
  """Compiles .fbs files using the flatc compiler."""
  with _ensure_flatc_executable() as executable_flatc:
    _compile_with_flatc(executable_flatc)


def _create_package_stubs() -> None:
  """Creates package directories before setup() accesses them."""
  os.makedirs("schema/py/runtime/proto", exist_ok=True)
  with open("schema/py/runtime/__init__.py", "a"):
    pass
  with open("schema/py/runtime/proto/__init__.py", "a"):
    pass

  os.makedirs("schema/py/schema/core/litert/lm/schema", exist_ok=True)
  with open("schema/py/schema/__init__.py", "a"):
    pass
  with open("schema/py/schema/core/__init__.py", "a"):
    pass
  with open("schema/py/schema/core/litert/__init__.py", "a"):
    pass
  with open("schema/py/schema/core/litert/lm/__init__.py", "a"):
    pass


class CustomBuildPy(_build_py):
  # Custom build_py command to compile protos and flatbuffers before building.
  def run(self) -> None:
    _ = self  # unused
    _create_package_stubs()
    compile_protos()
    compile_flatbuffers()
    super().run()


class CustomDevelop(_develop):
  # Custom develop command to compile protos and flatbuffers before building.
  def run(self) -> None:
    _ = self  # unused
    _create_package_stubs()
    compile_protos()
    compile_flatbuffers()
    super().run()


class CustomSdist(_sdist):
  # Custom sdist command to compile protos and flatbuffers before building.
  def run(self) -> None:
    _ = self  # unused
    _create_package_stubs()
    compile_protos()
    compile_flatbuffers()
    super().run()


class CustomEggInfo(_egg_info):
  # Custom egg_info command to create package
  # stubs so setuptools package detection won't fail.
  def run(self) -> None:
    _ = self  # unused
    _create_package_stubs()
    super().run()


def cleanup_generated() -> None:
  # Keep generated files for editable installs so they can be imported.
  is_editable = any("develop" in arg or "editable" in arg for arg in sys.argv)
  if not is_editable:
    shutil.rmtree("schema/py/runtime", ignore_errors=True)
    # Be careful to only remove schema/py/schema/core and related generated
    # files, actually schema/py/schema contains generated __init__ and core/
    shutil.rmtree("schema/py/schema", ignore_errors=True)

atexit.register(cleanup_generated)


setuptools.setup(
    name="litertlm-builder",
    version="0.0.1",
    author="Litert-lm Authors",
    description="LiteRT LM Builder library",
    long_description=(
        "Python tools for building, inspecting,"
        "and converting LiteRT-LM (.litertlm) files"
    ),
    long_description_content_type="text/markdown",
    url="https://github.com/google-ai-edge/LiteRT-LM.git",
    package_dir={
        "litert_lm": "schema/py",
        "litert_lm.runtime.proto": "schema/py/runtime/proto",
        "litert_lm.schema": "schema",
        "litert_lm.schema.py": "schema/py",
        "litert_lm.schema.core": "schema/py/schema/core",
        "litert": "schema/py/schema/core/litert",
        "litert.lm": "schema/py/schema/core/litert/lm",
        "litert.lm.schema": "schema/py/schema/core/litert/lm/schema",
    },
    packages=[
        "litert_lm",
        "litert_lm.runtime",
        "litert_lm.runtime.proto",
        "litert_lm.schema",
        "litert_lm.schema.py",
        "litert_lm.schema.core",
        "litert",
        "litert.lm",
        "litert.lm.schema",
    ],
    include_package_data=True,
    package_data={
        "litert_lm.runtime.proto": ["*.py"],
        "litert.lm.schema": ["*.py"],
        "litert_lm.schema.core": ["*.py"],
    },
    data_files=[
        ("runtime/proto", glob.glob("runtime/proto/*.proto")),
        ("schema/core", glob.glob("schema/core/*.fbs")),
    ],
    cmdclass={
        "build_py": CustomBuildPy,
        "develop": CustomDevelop,
        "sdist": CustomSdist,
        "egg_info": CustomEggInfo,
    },
    install_requires=[
        "protobuf",
        "flatbuffers",
        "absl-py",
        "tomli",
    ],
    entry_points={
        "console_scripts": [
            "litertlm-builder=litert_lm.litertlm_builder_cli:main",
            "litertlm-peek=litert_lm.litertlm_peek_main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
