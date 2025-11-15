#include "uci.hpp"
#include "move.hpp"
#include "movegen.hpp"
#include "nn_infer.hpp"

#include <iostream>
#include <sstream>

namespace chess {

UCI::UCI()
    : stBoard_(),
      ptrSearch_(new Search()),
      blnShouldQuit_(false) {}

UCI::~UCI() = default;

void UCI::cmd_uci() {
    std::cout << "id name ChessHacksNNEngine\n";
    std::cout << "id author Austin\n";
    std::cout << "uciok\n";
    std::cout.flush();
}

void UCI::cmd_isready() {
    std::cout << "readyok\n";
    std::cout.flush();
}

void UCI::cmd_ucinewgame() {
    stBoard_.setFromFEN("startpos");
}

void UCI::cmd_position(const std::string &strArgs) {
    std::istringstream iss(strArgs);
    std::string strToken;
    iss >> strToken;

    if (strToken == "startpos") {
        stBoard_.setFromFEN("startpos");
        if (iss >> strToken) {
            if (strToken == "moves") {
                std::string strMove;
                while (iss >> strMove) {
                    Move stMove = Move::from_uci(strMove);
                    apply_move(stBoard_, stMove);
                }
            }
        }
    } else if (strToken == "fen") {
        std::string strFENPart, strFENFull;
        int intParts = 0;
        while (intParts < 6 && iss >> strFENPart) {
            if (!strFENFull.empty()) strFENFull += " ";
            strFENFull += strFENPart;
            ++intParts;
        }
        if (intParts == 6) {
            stBoard_.setFromFEN(strFENFull);
        }

        if (iss >> strToken) {
            if (strToken == "moves") {
                std::string strMove;
                while (iss >> strMove) {
                    Move stMove = Move::from_uci(strMove);
                    apply_move(stBoard_, stMove);
                }
            }
        }
    }
}

void UCI::cmd_go(const std::string &strArgs) {
    std::istringstream iss(strArgs);
    std::string strToken;

    int intDepth = 4;
    int64_t intMoveTimeMs = 1000;

    int64_t intWTime = -1;
    int64_t intBTime = -1;

    while (iss >> strToken) {
        if (strToken == "depth") {
            iss >> intDepth;
        } else if (strToken == "movetime") {
            iss >> intMoveTimeMs;
        } else if (strToken == "wtime") {
            iss >> intWTime;
        } else if (strToken == "btime") {
            iss >> intBTime;
        }
    }

    if (intWTime >= 0 && intBTime >= 0) {
        intMoveTimeMs = (stBoard_.clrSideToMove == COLOR_WHITE) ? intWTime / 30 : intBTime / 30;
        if (intMoveTimeMs < 50) intMoveTimeMs = 50;
    }

    SearchResult stRes = ptrSearch_->search(stBoard_, intDepth, intMoveTimeMs);
    std::string strBest = stRes.stBestMove.to_uci();
    std::cout << "bestmove " << strBest << "\n";
    std::cout.flush();
}

void UCI::cmd_stop() {
    if (ptrSearch_) ptrSearch_->stop();
}

void UCI::cmd_quit() {
    blnShouldQuit_ = true;
}

void UCI::cmd_setoption(const std::string &/*strArgs*/) {
    // No options yet
}

void UCI::handle_line(const std::string &strLine) {
    std::istringstream iss(strLine);
    std::string strCmd;
    iss >> strCmd;

    if (strCmd == "uci") {
        cmd_uci();
    } else if (strCmd == "isready") {
        cmd_isready();
    } else if (strCmd == "ucinewgame") {
        cmd_ucinewgame();
    } else if (strCmd == "position") {
        std::string strRest;
        std::getline(iss, strRest);
        if (!strRest.empty() && strRest[0] == ' ') strRest.erase(0, 1);
        cmd_position(strRest);
    } else if (strCmd == "go") {
        std::string strRest;
        std::getline(iss, strRest);
        if (!strRest.empty() && strRest[0] == ' ') strRest.erase(0, 1);
        cmd_go(strRest);
    } else if (strCmd == "stop") {
        cmd_stop();
    } else if (strCmd == "quit") {
        cmd_quit();
    } else if (strCmd == "setoption") {
        std::string strRest;
        std::getline(iss, strRest);
        if (!strRest.empty() && strRest[0] == ' ') strRest.erase(0, 1);
        cmd_setoption(strRest);
    } else if (strCmd == "d") {
        std::cout << stBoard_.toString();
        std::cout.flush();
    }
}

void UCI::run() {
    std::string strLine;
    while (!blnShouldQuit_ && std::getline(std::cin, strLine)) {
        if (strLine.empty()) continue;
        handle_line(strLine);
    }
}

} // namespace chess
