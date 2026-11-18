"""Собирает Транскрибация.app — окно приложения поверх этой папки проекта.

    .venv/bin/python scripts/build_mac_app.py            # в ~/Applications
    .venv/bin/python scripts/build_mac_app.py --dest DIR

Приложение не самодостаточное: путь к проекту зашит при сборке, модели и
библиотеки берутся из .venv. Перенесли или переименовали папку — пересоберите.

Внутри бандла лежит копия интерпретатора Python.framework. Если запускать
.venv/bin/python напрямую, macOS считает процесс приложением «Python» с ракетой
в Dock. Копия внутри Contents/MacOS делает главным бандлом наш, а
__PYVENV_LAUNCHER__ направляет её в .venv — так же устроен сам bin/python3
во фреймворковой сборке.
"""

import argparse
import plistlib
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_NAME = "Транскрибация"
BUNDLE_ID = "local.transcriber"
EXECUTABLE = "transcriber"
LSREGISTER = (
    "/System/Library/Frameworks/CoreServices.framework/Frameworks/"
    "LaunchServices.framework/Support/lsregister"
)


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)


def framework_python() -> Path:
    path = Path(sys.base_prefix) / "Resources/Python.app/Contents/MacOS/Python"
    if not path.is_file():
        fail(
            f"Не найден {path}. Сборка рассчитана на Python с python.org "
            "(Python.framework); у Homebrew-Python и pyenv он лежит иначе."
        )
    return path


def version() -> str:
    try:
        import tomllib  # 3.11+
    except ModuleNotFoundError:
        return "0.1.0"
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]


def draw_icon(png: Path) -> None:
    """Синяя плашка в цвет --accent интерфейса и белая звуковая волна."""
    import AppKit

    size = 1024
    rep = AppKit.NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
        None, size, size, 8, 4, True, False, AppKit.NSDeviceRGBColorSpace, 0, 0
    )
    ctx = AppKit.NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep)
    AppKit.NSGraphicsContext.saveGraphicsState()
    AppKit.NSGraphicsContext.setCurrentContext_(ctx)

    # Сетка значков macOS: плашка 824×824 с отступом 100, скругление ~185.
    tile = AppKit.NSMakeRect(100, 100, 824, 824)
    path = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(tile, 185, 185)

    shadow = AppKit.NSShadow.alloc().init()
    shadow.setShadowOffset_((0, -10))
    shadow.setShadowBlurRadius_(28)
    shadow.setShadowColor_(AppKit.NSColor.colorWithWhite_alpha_(0, 0.3))
    AppKit.NSGraphicsContext.saveGraphicsState()
    shadow.set()
    AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(0.18, 0.44, 0.93, 1).setFill()
    path.fill()
    AppKit.NSGraphicsContext.restoreGraphicsState()

    gradient = AppKit.NSGradient.alloc().initWithStartingColor_endingColor_(
        AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(0.12, 0.33, 0.82, 1),
        AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(0.33, 0.58, 1.0, 1),
    )
    gradient.drawInBezierPath_angle_(path, 90)

    config = AppKit.NSImageSymbolConfiguration.configurationWithPointSize_weight_(
        420, AppKit.NSFontWeightSemibold
    ).configurationByApplyingConfiguration_(
        AppKit.NSImageSymbolConfiguration.configurationWithPaletteColors_([AppKit.NSColor.whiteColor()])
    )
    symbol = AppKit.NSImage.imageWithSystemSymbolName_accessibilityDescription_(
        "waveform", None
    ).imageWithSymbolConfiguration_(config)
    w, h = symbol.size()
    symbol.drawInRect_(AppKit.NSMakeRect((size - w) / 2, (size - h) / 2, w, h))

    AppKit.NSGraphicsContext.restoreGraphicsState()
    data = rep.representationUsingType_properties_(AppKit.NSBitmapImageFileTypePNG, {})
    data.writeToFile_atomically_(str(png), True)


def make_icns(dest: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        master = Path(tmp) / "icon.png"
        draw_icon(master)
        iconset = Path(tmp) / "AppIcon.iconset"
        iconset.mkdir()
        for base in (16, 32, 128, 256, 512):
            for scale in (1, 2):
                px = base * scale
                name = f"icon_{base}x{base}{'@2x' if scale == 2 else ''}.png"
                subprocess.run(
                    ["sips", "-z", str(px), str(px), str(master), "--out", str(iconset / name)],
                    check=True,
                    capture_output=True,
                )
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(dest)], check=True)


def launcher_script() -> str:
    project = shlex.quote(str(ROOT))
    # Из Finder приложение стартует с PATH=/usr/bin:/bin:/usr/sbin:/sbin — без
    # Homebrew, и ffmpeg не находится. Берём каталог, где он виден сейчас.
    extra_path = ["/opt/homebrew/bin", "/usr/local/bin"]
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg and str(Path(ffmpeg).parent) not in extra_path:
        extra_path.insert(0, str(Path(ffmpeg).parent))

    return f"""#!/bin/sh
# Сгенерировано scripts/build_mac_app.py. Путь к проекту зашит при сборке.
PROJECT={project}
LOG="$HOME/Library/Logs/{APP_NAME}.log"

# Журнал не растёт бесконечно: на старте больше 5 МБ — откладываем в .1.
if [ -f "$LOG" ] && [ "$(stat -f%z "$LOG")" -gt 5000000 ]; then mv -f "$LOG" "$LOG.1"; fi
exec >>"$LOG" 2>&1
echo "=== запуск $(date '+%Y-%m-%d %H:%M:%S')"

if [ ! -x "$PROJECT/.venv/bin/python" ]; then
  osascript -e 'display alert "{APP_NAME}" message "Не найдена папка проекта или её окружение .venv. Если папку переносили или переименовывали — пересоберите приложение: .venv/bin/python scripts/build_mac_app.py"'
  exit 1
fi

cd "$PROJECT" || exit 1
export PATH={shlex.quote(":".join(extra_path))}:"$PATH"
export TRANSCRIBER_LOG="$LOG"
export __PYVENV_LAUNCHER__="$PROJECT/.venv/bin/python"
exec "$(dirname "$0")/python" -m backend.desktop
"""


def build(dest_dir: Path) -> Path:
    app = dest_dir / f"{APP_NAME}.app"
    if app.exists():
        try:
            info = plistlib.loads((app / "Contents/Info.plist").read_bytes())
        except (OSError, plistlib.InvalidFileException):
            info = {}
        if info.get("CFBundleIdentifier") != BUNDLE_ID:
            fail(f"{app} уже существует и собран не этим скриптом — не трогаю.")
        shutil.rmtree(app)

    macos = app / "Contents/MacOS"
    resources = app / "Contents/Resources"
    macos.mkdir(parents=True)
    resources.mkdir()

    shutil.copy2(framework_python(), macos / "python")
    launcher = macos / EXECUTABLE
    launcher.write_text(launcher_script(), encoding="utf-8")
    launcher.chmod(0o755)
    make_icns(resources / "AppIcon.icns")

    (app / "Contents/Info.plist").write_bytes(
        plistlib.dumps(
            {
                "CFBundleName": APP_NAME,
                "CFBundleDisplayName": APP_NAME,
                "CFBundleIdentifier": BUNDLE_ID,
                "CFBundleExecutable": EXECUTABLE,
                "CFBundleIconFile": "AppIcon",
                "CFBundlePackageType": "APPL",
                "CFBundleShortVersionString": version(),
                "CFBundleVersion": version(),
                "LSApplicationCategoryType": "public.app-category.productivity",
                "LSMinimumSystemVersion": "12.0",
                "NSHighResolutionCapable": True,
            }
        )
    )

    # Копия интерпретатора несёт подпись python.org, привязанную к чужому бандлу.
    # Переподписываем ad hoc: сборка локальная, карантина нет, Gatekeeper не спросит.
    for target in (macos / "python", app):
        subprocess.run(["codesign", "--force", "--sign", "-", str(target)], check=True, capture_output=True)

    # Иначе Finder и Dock показывают закэшированную старую иконку.
    subprocess.run([LSREGISTER, "-f", str(app)], check=False, capture_output=True)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=Path.home() / "Applications")
    args = parser.parse_args()

    if sys.platform != "darwin":
        fail("Приложение собирается только на macOS.")
    try:
        import webview  # noqa: F401
    except ModuleNotFoundError:
        fail("Нет pywebview. Поставьте: .venv/bin/python -m pip install -e '.[mac]'")

    args.dest.mkdir(parents=True, exist_ok=True)
    app = build(args.dest)
    print(f"Готово: {app}")
    print("Журнал: ~/Library/Logs/Транскрибация.log")


if __name__ == "__main__":
    main()
