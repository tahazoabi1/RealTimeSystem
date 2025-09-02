#include <boost/throw_exception.hpp>
#include <boost/exception/exception.hpp>
#include <stdexcept>

namespace boost {
    void throw_exception(std::exception const & e) {
        throw e;
    }
}