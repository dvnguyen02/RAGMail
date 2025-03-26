from app import main

if __name__ == "__main__":
    main()
# to this one. The difference is that the first one imports sys and calls sys.exit(main()) while the second one calls main() directly. 
# The sys.exit() function is used to exit the program with a status code. In this case, 
# it is not necessary to use sys.exit() because the main() function is the last statement in the script and will exit the program when it finishes executing. 
# Therefore, the second snippet is more concise and does the same job as the first one.