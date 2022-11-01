#include <stdio.h>
#include <dirent.h>
//https://stackoverflow.com/questions/1427032/fast-linux-file_count-for-a-large-number-of-files
//Christopher Schultz
//Apache 2.0

int main(int argc, char *argv[]) {
    DIR *dir;
    struct dirent *ent;
    long count = 0;

    dir = opendir(argv[1]);

    while((ent = readdir(dir)))
        ++count;

    closedir(dir);

    printf("%s contains %ld files\n", argv[1], count);

    return 0;
}
