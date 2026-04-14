#!/bin/bash
# continue setup from after azure AI search (already created)

set -e

RESOURCE_GROUP="metis"
LOCATION="westeurope"
STORAGE_NAME="metisstore42"
FUNCTION_APP="metis-automata"
OPENAI_NAME="metis-openai"

echo "creating storage account..."
az storage account create \
  --name $STORAGE_NAME \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku Standard_LRS

echo "creating blob containers..."
az storage container create \
  --name ingest \
  --account-name $STORAGE_NAME

az storage container create \
  --name pending \
  --account-name $STORAGE_NAME

echo "creating azure openai resource..."
az cognitiveservices account create \
  --name $OPENAI_NAME \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --kind OpenAI \
  --sku S0

echo "deploying openai models..."
az cognitiveservices account deployment create \
  --name $OPENAI_NAME \
  --resource-group $RESOURCE_GROUP \
  --deployment-name gpt-4o \
  --model-name gpt-4o \
  --model-version "2024-11-20" \
  --model-format OpenAI \
  --sku-capacity 10 \
  --sku-name GlobalStandard

az cognitiveservices account deployment create \
  --name $OPENAI_NAME \
  --resource-group $RESOURCE_GROUP \
  --deployment-name text-embedding-3-small \
  --model-name text-embedding-3-small \
  --model-version "1" \
  --model-format OpenAI \
  --sku-capacity 10 \
  --sku-name GlobalStandard

echo "creating function app..."
az functionapp create \
  --name $FUNCTION_APP \
  --resource-group $RESOURCE_GROUP \
  --storage-account $STORAGE_NAME \
  --consumption-plan-location $LOCATION \
  --runtime python \
  --runtime-version 3.11 \
  --functions-version 4 \
  --os-type Linux

echo ""
echo "done. fetching keys..."
echo ""

echo "search admin key:"
az search admin-key show \
  --service-name metis-search \
  --resource-group $RESOURCE_GROUP \
  --query primaryKey -o tsv

echo ""
echo "storage connection string:"
az storage account show-connection-string \
  --name $STORAGE_NAME \
  --resource-group $RESOURCE_GROUP \
  --query connectionString -o tsv

echo ""
echo "openai endpoint:"
az cognitiveservices account show \
  --name $OPENAI_NAME \
  --resource-group $RESOURCE_GROUP \
  --query properties.endpoint -o tsv

echo ""
echo "openai key:"
az cognitiveservices account keys list \
  --name $OPENAI_NAME \
  --resource-group $RESOURCE_GROUP \
  --query key1 -o tsv

echo ""
echo "save these keys."
