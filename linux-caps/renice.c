#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>

int getpriority(int which, id_t who);
int setpriority(int which, id_t who, int prio);

int main(int argc, const char * argv[]) {
    pid_t pid = getpid();
    id_t who = 2;
    errno = ENOENT;
    int prio = getpriority(PRIO_USER, who);
    if (errno != ENOENT) {
        perror("getpriority failed!");
        return 0;
    }
    printf("getpriority PRIO_USER %d returns %d\n", who, prio);

    errno = ENOENT;
    setpriority(PRIO_USER, who, prio);
    if (errno != ENOENT) {
        printf("setpriority PRIO_USER %d to %d failed!\n", who, prio);
        perror("\t");
        printf("run following command to fix this:\n");
        printf(" sudo setcap cap_sys_nice=eip %s\n", argv[0]);
        return 0;
    }
    printf("setpriority PRIO_USER %d to %d Success!\n", who, prio);
    
    printf(" check capabilities by:\n cat /proc/%d/status | grep Cap\n", pid);
    printf(" then press return to exit!");
    getchar();
    return 0;
}
