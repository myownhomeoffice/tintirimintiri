void display_usage() {
    printf("DUMMY [-h]\n");
    printf("-h\t\tDisplay this help message.\n");
}

void dump_pchar_array(char* envp[]) {
    int index = 0;
    while(envp[index] != 0) {
        printf("envp[%d] = %s", index, envp[index]);
        ++index;
    }
}

int main(int argc, char* argv[], char* envp[]) {
    if(argc <= 1) {
        display_usage();
        return 0;
    }
    // nqkakav comment
    return 0;
}
