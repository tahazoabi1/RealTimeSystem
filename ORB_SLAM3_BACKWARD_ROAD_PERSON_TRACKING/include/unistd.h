#ifndef UNISTD_H_WINDOWS_COMPAT
#define UNISTD_H_WINDOWS_COMPAT

#ifdef _WIN32

#include <io.h>
#include <process.h>
#include <direct.h>
#include <windows.h>

#define ssize_t long
#define usleep(x) Sleep((x)/1000)

#ifndef R_OK
#define R_OK 4
#define W_OK 2
#define X_OK 1
#define F_OK 0
#endif

// Use function wrappers instead of macros to avoid conflicts with Boost serialization
static inline int unix_access(const char* path, int mode) { return _access(path, mode); }
static inline char* unix_getcwd(char* buf, size_t size) { return _getcwd(buf, (int)size); }
static inline int unix_chdir(const char* path) { return _chdir(path); }

// Only define these macros if Boost serialization headers haven't been included
#ifndef BOOST_SERIALIZATION_ACCESS_HPP
#define access _access
#endif
#define getcwd _getcwd  
#define chdir _chdir

#else
#include <unistd.h>
#endif

#endif