void display_usage() {
    printf("DUMMY [-h]\n");
    printf("-h\t\tDisplay this help message.\n");
}

int main(int argc, char* argv[]) {
    if(argc <= 1) {
        display_usage();
        return 0;
    }
    return 0;
}
