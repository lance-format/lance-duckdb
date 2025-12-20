@echo off
setlocal

if not "%PROTOC_REAL%"=="" (
  if exist "%PROTOC_REAL%" (
    "%PROTOC_REAL%" %*
    exit /b %errorlevel%
  )
)

set "VCPKG_ROOT=%VCPKG_ROOT%"
if "%VCPKG_ROOT%"=="" goto :fallback

set "TRIPLET=%VCPKG_HOST_TRIPLET%"
if "%TRIPLET%"=="" set "TRIPLET=%VCPKG_TARGET_TRIPLET%"
if "%TRIPLET%"=="" goto :fallback

set "INSTALLED=%VCPKG_INSTALLED_DIR%"
if "%INSTALLED%"=="" set "INSTALLED=%VCPKG_ROOT%\\installed"

set "BASE1=%INSTALLED%\\%TRIPLET%\\tools\\protobuf"
for %%F in ("%BASE1%\\protoc.exe" "%BASE1%\\protoc-*.exe") do (
  if exist "%%~fF" (
    "%%~fF" %*
    exit /b %errorlevel%
  )
)

set "BASE2=%VCPKG_ROOT%\\packages\\protobuf_%TRIPLET%\\tools\\protobuf"
for %%F in ("%BASE2%\\protoc.exe" "%BASE2%\\protoc-*.exe") do (
  if exist "%%~fF" (
    "%%~fF" %*
    exit /b %errorlevel%
  )
)

:fallback
where protoc >nul 2>nul
if %errorlevel%==0 (
  protoc %*
  exit /b %errorlevel%
)

echo protoc_wrapper: protoc not found 1>&2
exit /b 127
