cd ./src || exit
gcloud config set project angelhack-finhack-2024
gcloud builds submit --tag gcr.io/angelhack-finhack-2024/finhack-sme-loan-api