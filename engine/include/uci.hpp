#pragma once

#include "board.hpp"
#include "search.hpp"
#include <string>
#include <memory>

namespace chess {

class UCI {
public:
    UCI();
    ~UCI();

    void run();

private:
    void handle_line(const std::string &strLine);

    void cmd_uci();
    void cmd_isready();
    void cmd_ucinewgame();
    void cmd_position(const std::string &strArgs);
    void cmd_go(const std::string &strArgs);
    void cmd_stop();
    void cmd_quit();
    void cmd_setoption(const std::string &strArgs);

    Board                      stBoard_;
    std::unique_ptr<Search>    ptrSearch_;
    bool                       blnShouldQuit_;
};

} // namespace chess
