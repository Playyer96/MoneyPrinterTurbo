@echo off
setlocal enabledelayedexpansion

set "INPUT=%~1"
if not defined INPUT set "INPUT=input.mp4"

if not exist "%INPUT%" (
    echo File "%INPUT%" not found.
    echo Drag and drop a video onto this script.
    pause
    exit /b 1
)

echo Reading chapters from "%INPUT%"...

set "INDEX=0"

rem Extract start_time and end_time pairs on alternating lines
set "START="
for /f "tokens=1 delims=" %%A in ('ffprobe -v error -show_entries "chapter=start_time,end_time" -of "default=noprint_wrappers=1:nokey=1" "%INPUT%"') do (
    if not defined START (
        set "START=%%A"
    ) else (
        set "END=%%A"
        set /a INDEX+=1

        rem Format index as 2 digits (01, 02, etc.)
        set "NUM=0!INDEX!"
        set "NUM=!NUM:~-2!"

        echo Extracting Chapter !NUM! [!START! s to !END! s]...
        ffmpeg -y -ss !START! -to !END! -i "%INPUT%" -c copy -map_metadata -1 "chapter_!NUM!.mp4"

        rem Reset START for the next chapter pair
        set "START="
    )
)

if !INDEX!==0 (
    echo No chapters were found in this video file.
    pause
    exit /b 1
)

echo.
echo Done! Extracted !INDEX! chapter clips.
pause