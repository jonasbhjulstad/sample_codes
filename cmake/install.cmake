# Introduce variables:
# * CMAKE_INSTALL_LIBDIR
# * CMAKE_INSTALL_BINDIR
# * CMAKE_INSTALL_INCLUDEDIR
include(GNUInstallDirs)

packageProject(
  # the name of the target to export
  NAME ${PROJECT_NAME}
  # the version of the target to export
  VERSION ${PROJECT_VERSION}
  # a temporary directory to create the config files
  BINARY_DIR ${PROJECT_BINARY_DIR}
  # location of the target's public headers
  INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include
  # should match the target's INSTALL_INTERFACE include directory
  INCLUDE_DESTINATION include/
  # (optional) option to install only header files with matching pattern
  INCLUDE_HEADER_PATTERN "*.hpp"
  # semicolon separated list of the project's dependencies
  DEPENDENCIES "Static_RNG 1.0.0; cppitertools 2.1.0"
  # (optional) create a header containing the version info
  # Note: that the path to headers should be lowercase
  VERSION_HEADER "${PROJECT_NAME}/version.h"
  # (optional) create a export header using GenerateExportHeader module
  EXPORT_HEADER "${PROJECT_NAME}/export.h"
  # (optional) install your library with a namespace (Note: do NOT add extra '::')
  NAMESPACE ${PROJECT_NAMESPACE}
  # (optional) define the project's version compatibility, defaults to `AnyNewerVersion`
  # supported values: `AnyNewerVersion|SameMajorVersion|SameMinorVersion|ExactVersion`
  COMPATIBILITY AnyNewerVersion
  # (optional) option to disable the versioning of install destinations
  DISABLE_VERSION_SUFFIX YES
  # (optional) option to ignore target architecture for package resolution
  # defaults to YES for header only (i.e. INTERFACE) libraries
  ARCH_INDEPENDENT YES
)