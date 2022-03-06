import platform
import subprocess


def get_java_loc():
    # Check OS to determine what command to run
    os_system = platform.system()
    if os_system == "Windows":
        command = "where java"
    elif os_system in ("Linux", "Darwin"):
        command = "which java"
    else:
        raise NotImplementedError(f"The {os_system} operating system has not been implemented yet.")

    # Run a shell script to locate the java executable
    location = subprocess.check_output(command, shell=True).strip()
    if not location:
        raise FileNotFoundError("Install java first!")
    return location

if __name__ == "__main__":
    java_loc = get_java_loc()
    print(java_loc)