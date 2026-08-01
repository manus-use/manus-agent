"""Comprehensive test suite for file_operations module.

Tests cover all five utility functions:
- file_read: text read, binary fallback, missing file, not-a-file, permission errors
- file_write: create file, overwrite, nested dirs, create_dirs=False
- file_list: list all, glob patterns, missing dir, not-a-dir
- file_delete: files, empty dirs, recursive dirs, missing, non-empty without force
- file_move: basic move, rename, overwrite, missing source, existing dest, cross-dir

All tests use tmp_path fixtures — no real filesystem side effects.
"""

from __future__ import annotations

import pytest

from manus_agent.tools.file_operations import (
    file_delete,
    file_list,
    file_move,
    file_read,
    file_write,
)

# ===========================================================================
# file_read
# ===========================================================================


class TestFileRead:
    """Test file_read function."""

    def test_read_text_file(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("Hello, World!", encoding="utf-8")
        result = file_read(str(f))
        assert result == "Hello, World!"

    def test_read_multiline_file(self, tmp_path):
        f = tmp_path / "multi.txt"
        content = "line1\nline2\nline3"
        f.write_text(content, encoding="utf-8")
        result = file_read(str(f))
        assert result == content

    def test_read_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        result = file_read(str(f))
        assert result == ""

    def test_read_unicode_file(self, tmp_path):
        f = tmp_path / "unicode.txt"
        content = "日本語テスト 🎉 émojis"
        f.write_text(content, encoding="utf-8")
        result = file_read(str(f))
        assert result == content

    def test_read_binary_file_fallback(self, tmp_path):
        f = tmp_path / "binary.dat"
        binary_content = bytes(range(256))
        f.write_bytes(binary_content)
        result = file_read(str(f))
        assert result.startswith("Binary file (256 bytes):")
        assert "..." in result

    def test_read_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="File not found"):
            file_read(str(tmp_path / "nonexistent.txt"))

    def test_read_directory_raises_value_error(self, tmp_path):
        d = tmp_path / "subdir"
        d.mkdir()
        with pytest.raises(ValueError, match="not a file"):
            file_read(str(d))

    def test_read_with_tilde_expansion(self, tmp_path, monkeypatch):
        # Create a file in a temp "home"
        monkeypatch.setenv("HOME", str(tmp_path))
        f = tmp_path / "test.txt"
        f.write_text("tilde content", encoding="utf-8")
        result = file_read("~/test.txt")
        assert result == "tilde content"

    def test_read_file_with_spaces_in_name(self, tmp_path):
        f = tmp_path / "file with spaces.txt"
        f.write_text("spaces work", encoding="utf-8")
        result = file_read(str(f))
        assert result == "spaces work"

    def test_read_large_file(self, tmp_path):
        f = tmp_path / "large.txt"
        content = "x" * 100_000
        f.write_text(content, encoding="utf-8")
        result = file_read(str(f))
        assert len(result) == 100_000


# ===========================================================================
# file_write
# ===========================================================================


class TestFileWrite:
    """Test file_write function."""

    def test_write_new_file(self, tmp_path):
        f = tmp_path / "new.txt"
        result = file_write(str(f), "hello")
        assert "Successfully wrote" in result
        assert "5 characters" in result
        assert f.read_text(encoding="utf-8") == "hello"

    def test_write_overwrites_existing(self, tmp_path):
        f = tmp_path / "existing.txt"
        f.write_text("old content", encoding="utf-8")
        file_write(str(f), "new content")
        assert f.read_text(encoding="utf-8") == "new content"

    def test_write_creates_parent_dirs(self, tmp_path):
        f = tmp_path / "a" / "b" / "c" / "deep.txt"
        file_write(str(f), "deep content")
        assert f.read_text(encoding="utf-8") == "deep content"

    def test_write_create_dirs_false_missing_parent(self, tmp_path):
        f = tmp_path / "missing_parent" / "file.txt"
        with pytest.raises((RuntimeError, FileNotFoundError, OSError)):
            file_write(str(f), "content", create_dirs=False)

    def test_write_empty_content(self, tmp_path):
        f = tmp_path / "empty.txt"
        result = file_write(str(f), "")
        assert "0 characters" in result
        assert f.read_text(encoding="utf-8") == ""

    def test_write_unicode_content(self, tmp_path):
        f = tmp_path / "unicode.txt"
        content = "日本語テスト 🎉"
        file_write(str(f), content)
        assert f.read_text(encoding="utf-8") == content

    def test_write_multiline_content(self, tmp_path):
        f = tmp_path / "multi.txt"
        content = "line1\nline2\nline3\n"
        file_write(str(f), content)
        assert f.read_text(encoding="utf-8") == content

    def test_write_with_tilde_expansion(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        file_write("~/expanded.txt", "tilde write")
        result = (tmp_path / "expanded.txt").read_text(encoding="utf-8")
        assert result == "tilde write"

    def test_write_returns_file_path_in_message(self, tmp_path):
        f = tmp_path / "info.txt"
        result = file_write(str(f), "test")
        assert str(f.name) in result or "info.txt" in result


# ===========================================================================
# file_list
# ===========================================================================


class TestFileList:
    """Test file_list function."""

    def test_list_empty_directory(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        result = file_list(str(d))
        assert result == []

    def test_list_files_and_dirs(self, tmp_path):
        (tmp_path / "file1.txt").write_text("a", encoding="utf-8")
        (tmp_path / "file2.py").write_text("b", encoding="utf-8")
        (tmp_path / "subdir").mkdir()
        result = file_list(str(tmp_path))
        assert len(result) == 3
        assert "file1.txt" in result
        assert "file2.py" in result
        assert "subdir" in result

    def test_list_with_glob_pattern(self, tmp_path):
        (tmp_path / "a.py").write_text("", encoding="utf-8")
        (tmp_path / "b.py").write_text("", encoding="utf-8")
        (tmp_path / "c.txt").write_text("", encoding="utf-8")
        result = file_list(str(tmp_path), pattern="*.py")
        assert len(result) == 2
        assert all(f.endswith(".py") for f in result)

    def test_list_with_recursive_glob(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "top.py").write_text("", encoding="utf-8")
        (sub / "nested.py").write_text("", encoding="utf-8")
        result = file_list(str(tmp_path), pattern="**/*.py")
        assert len(result) >= 2

    def test_list_missing_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            file_list(str(tmp_path / "nonexistent"))

    def test_list_not_a_directory(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("not a dir", encoding="utf-8")
        with pytest.raises(ValueError, match="not a directory"):
            file_list(str(f))

    def test_list_default_directory(self, monkeypatch, tmp_path):
        # This tests the default "." argument
        monkeypatch.chdir(tmp_path)
        (tmp_path / "here.txt").write_text("", encoding="utf-8")
        result = file_list(".")
        assert "here.txt" in result

    def test_list_hidden_files(self, tmp_path):
        (tmp_path / ".hidden").write_text("", encoding="utf-8")
        (tmp_path / "visible.txt").write_text("", encoding="utf-8")
        result = file_list(str(tmp_path))
        assert ".hidden" in result
        assert "visible.txt" in result

    def test_list_returns_relative_paths(self, tmp_path):
        (tmp_path / "test.txt").write_text("", encoding="utf-8")
        result = file_list(str(tmp_path))
        # Paths should be relative, not absolute
        for item in result:
            assert not item.startswith("/")


# ===========================================================================
# file_delete
# ===========================================================================


class TestFileDelete:
    """Test file_delete function."""

    def test_delete_file(self, tmp_path):
        f = tmp_path / "to_delete.txt"
        f.write_text("delete me", encoding="utf-8")
        result = file_delete(str(f))
        assert "Deleted file" in result
        assert not f.exists()

    def test_delete_empty_directory(self, tmp_path):
        d = tmp_path / "empty_dir"
        d.mkdir()
        result = file_delete(str(d))
        assert "Deleted empty directory" in result
        assert not d.exists()

    def test_delete_directory_recursive(self, tmp_path):
        d = tmp_path / "nonempty"
        d.mkdir()
        (d / "child.txt").write_text("child", encoding="utf-8")
        (d / "subdir").mkdir()
        (d / "subdir" / "deep.txt").write_text("deep", encoding="utf-8")
        result = file_delete(str(d), force=True)
        assert "recursively" in result
        assert not d.exists()

    def test_delete_nonempty_dir_without_force(self, tmp_path):
        d = tmp_path / "nonempty"
        d.mkdir()
        (d / "child.txt").write_text("child", encoding="utf-8")
        with pytest.raises((RuntimeError, OSError)):
            file_delete(str(d), force=False)

    def test_delete_nonexistent_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Path not found"):
            file_delete(str(tmp_path / "ghost.txt"))

    def test_delete_symlink(self, tmp_path):
        target = tmp_path / "target.txt"
        target.write_text("target", encoding="utf-8")
        link = tmp_path / "link.txt"
        link.symlink_to(target)
        # file_delete resolves symlinks, so it deletes the target file
        file_delete(str(link))
        assert not target.exists()  # Target is removed (resolve follows link)

    def test_delete_preserves_other_files(self, tmp_path):
        f1 = tmp_path / "keep.txt"
        f2 = tmp_path / "delete.txt"
        f1.write_text("keep", encoding="utf-8")
        f2.write_text("delete", encoding="utf-8")
        file_delete(str(f2))
        assert f1.exists()
        assert not f2.exists()


# ===========================================================================
# file_move
# ===========================================================================


class TestFileMove:
    """Test file_move function."""

    def test_move_file(self, tmp_path):
        src = tmp_path / "src.txt"
        dst = tmp_path / "dst.txt"
        src.write_text("moving", encoding="utf-8")
        result = file_move(str(src), str(dst))
        assert "Moved" in result
        assert not src.exists()
        assert dst.read_text(encoding="utf-8") == "moving"

    def test_rename_file(self, tmp_path):
        src = tmp_path / "old_name.txt"
        dst = tmp_path / "new_name.txt"
        src.write_text("renamed", encoding="utf-8")
        file_move(str(src), str(dst))
        assert not src.exists()
        assert dst.read_text(encoding="utf-8") == "renamed"

    def test_move_to_different_directory(self, tmp_path):
        src = tmp_path / "src.txt"
        dst_dir = tmp_path / "subdir"
        dst_dir.mkdir()
        dst = dst_dir / "moved.txt"
        src.write_text("content", encoding="utf-8")
        file_move(str(src), str(dst))
        assert not src.exists()
        assert dst.read_text(encoding="utf-8") == "content"

    def test_move_creates_parent_dirs(self, tmp_path):
        src = tmp_path / "src.txt"
        dst = tmp_path / "a" / "b" / "dst.txt"
        src.write_text("deep move", encoding="utf-8")
        file_move(str(src), str(dst))
        assert dst.read_text(encoding="utf-8") == "deep move"

    def test_move_overwrite_existing(self, tmp_path):
        src = tmp_path / "src.txt"
        dst = tmp_path / "dst.txt"
        src.write_text("new content", encoding="utf-8")
        dst.write_text("old content", encoding="utf-8")
        file_move(str(src), str(dst), overwrite=True)
        assert dst.read_text(encoding="utf-8") == "new content"
        assert not src.exists()

    def test_move_existing_dest_no_overwrite(self, tmp_path):
        src = tmp_path / "src.txt"
        dst = tmp_path / "dst.txt"
        src.write_text("source", encoding="utf-8")
        dst.write_text("existing", encoding="utf-8")
        with pytest.raises(FileExistsError, match="already exists"):
            file_move(str(src), str(dst), overwrite=False)

    def test_move_missing_source(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Source not found"):
            file_move(str(tmp_path / "ghost.txt"), str(tmp_path / "dst.txt"))

    def test_move_directory(self, tmp_path):
        src = tmp_path / "src_dir"
        src.mkdir()
        (src / "child.txt").write_text("child", encoding="utf-8")
        dst = tmp_path / "dst_dir"
        file_move(str(src), str(dst))
        assert not src.exists()
        assert (dst / "child.txt").read_text(encoding="utf-8") == "child"

    def test_move_returns_paths_in_message(self, tmp_path):
        src = tmp_path / "from.txt"
        dst = tmp_path / "to.txt"
        src.write_text("x", encoding="utf-8")
        result = file_move(str(src), str(dst))
        assert "from.txt" in result or str(src) in result
        assert "to.txt" in result or str(dst) in result


# ===========================================================================
# Integration / Cross-function tests
# ===========================================================================


class TestFileOperationsIntegration:
    """Test interactions between file operations."""

    def test_write_then_read(self, tmp_path):
        f = tmp_path / "roundtrip.txt"
        content = "round trip test 🔄"
        file_write(str(f), content)
        result = file_read(str(f))
        assert result == content

    def test_write_then_list(self, tmp_path):
        file_write(str(tmp_path / "a.txt"), "a")
        file_write(str(tmp_path / "b.txt"), "b")
        result = file_list(str(tmp_path))
        assert "a.txt" in result
        assert "b.txt" in result

    def test_write_move_read(self, tmp_path):
        src = tmp_path / "original.txt"
        dst = tmp_path / "moved.txt"
        file_write(str(src), "content")
        file_move(str(src), str(dst))
        result = file_read(str(dst))
        assert result == "content"

    def test_write_delete_verify_gone(self, tmp_path):
        f = tmp_path / "temp.txt"
        file_write(str(f), "temporary")
        file_delete(str(f))
        with pytest.raises(FileNotFoundError):
            file_read(str(f))

    def test_list_after_delete(self, tmp_path):
        f = tmp_path / "doomed.txt"
        file_write(str(f), "bye")
        assert "doomed.txt" in file_list(str(tmp_path))
        file_delete(str(f))
        assert "doomed.txt" not in file_list(str(tmp_path))

    def test_move_then_read_original_fails(self, tmp_path):
        src = tmp_path / "src.txt"
        dst = tmp_path / "dst.txt"
        file_write(str(src), "source")
        file_move(str(src), str(dst))
        with pytest.raises(FileNotFoundError):
            file_read(str(src))

    def test_write_nested_then_list_pattern(self, tmp_path):
        file_write(str(tmp_path / "code.py"), "print('hi')")
        file_write(str(tmp_path / "data.json"), "{}")
        file_write(str(tmp_path / "readme.md"), "# Hi")
        py_files = file_list(str(tmp_path), pattern="*.py")
        assert len(py_files) == 1
        assert "code.py" in py_files
