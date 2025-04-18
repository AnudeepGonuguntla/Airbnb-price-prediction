pipeline {
    agent any

    environment {
        // Define Python virtual environment path
        VENV = 'venv'
    }

    stages {
        stage('Clone Repository') {
            steps {
                git 'https://github.com/AnudeepGonuguntla/Airbnb-price-prediction'
            }
        }

        stage('Set Up Python') {
            steps {
                sh '''
                    python3 -m venv $VENV
                    . $VENV/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Run Python Script') {
            steps {
                sh '''
                    . $VENV/bin/activate
                    python jupyter nbconvert --to script notebook_name.ipynb

                '''
            }
        }

        stage('Clean Up') {
            steps {
                sh 'rm -rf $VENV'
            }
        }
    }

    post {
        failure {
            echo 'Pipeline failed!'
        }
        success {
            echo 'Pipeline executed successfully!'
        }
    }
}
