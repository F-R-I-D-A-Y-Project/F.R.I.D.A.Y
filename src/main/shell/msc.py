import subprocess, os, sys, pexpect, platform
class MSC:
    '''
        This class is responsible for the communication between the chatbot and the shell
    '''

    def __init__(self) -> None:
        self.platform = platform.system()
        # if 
        if self.platform == 'Windows':
            self.init_cmd = "cmd.exe"
        elif self.platform == 'Linux' or self.platform == 'Darwin':
            self.init_cmd = "bash"
        self.child_process: pexpect.spawn

    def send(self, message: str) -> None:
        '''
            This method sends a message to the shell and returns the output of the command.
        '''
        self.child_process.sendline(message)
        self.child_process.expect_exact("") #! find the correct way to isolate the output of the command
        return self.child_process.before.decode("utf-8")

    def __enter__(self) -> pexpect.spawn:
        self.child_process = pexpect.spawn(self.init_cmd)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.child_process.close()

    pass

def main():
    with MSC() as mcs:
        print(mcs.send("whoami"))
        print(mcs.send("whoami"))

if __name__ == "__main__":
    main()