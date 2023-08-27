import subprocess

def git_auto_commit(version, epoch):
    try:
        # Run 'git add .'
        subprocess.run(['git', 'add', '.'], check=True)

        # Generate commit message
        commit_message = f'[AUTO] training model v{version}, epoch {epoch}.'

        # Run 'git commit'
        subprocess.run(['git', 'commit', '-m', commit_message], check=True)

        # Run 'git push origin main'
        subprocess.run(['git', 'push', 'origin', 'main'], check=True)
        
        print("Git operations completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    git_auto_commit('version-manual', -1)