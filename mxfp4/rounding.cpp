#include <stdio.h>
#include <cfenv>
#include <cmath>


//============================
// rounds to the nearest value;
//   if the number falls midway it is rounded to the nearest value with an even (zero) least significant bit
//   which means it is rounded up 50% of the time.

//============================
// great explainations about roundTiesToEven:
//
// https://www.gnu.org/software/gawk/manual/html_node/Setting-the-rounding-mode.html

/**
-3.5 => -4.0
-2.5 => -2.0
-1.5 => -2.0
-0.5 => -0.0
 0.5 =>  0.0
 1.5 =>  2.0
 2.5 =>  2.0
 3.5 =>  4.0
 4.5 =>  4.0
 * 
*/
int main() {
    // by default TO NEAREST uses roundTiesToEven
    std::fesetround(FE_TONEAREST);

    // std::roundf always 四舍五入
    //    rounding halfway cases away from zero, regardless of the current rounding mode
    //
    // std::rint by-default roundTiesToEven
    //   using the current rounding mode.
    //
    //
    {
        float x = -4.5f;
        for (int i = 1; i < 10; i++) {
            x += 1.0;
            printf("%4.1f => %4.1f(rint)  %4.1f(roundf) \n", x, std::rint(x), std::roundf(x));
        }
    }
    return 0;
}
