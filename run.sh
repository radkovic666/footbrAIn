#!/bin/bash
set -e

# Configuration
WORK_DIR="/home/magilinux/footpredict"
ZIP_FILE="$WORK_DIR/player-scores.zip"
DB_FILE="$WORK_DIR/football_data.db"
URL="https://www.kaggle.com/api/v1/datasets/download/davidcariboo/player-scores"

# ASCII Art and Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}
  ______           _        _   _             _   
 |  ____|         | |      | | (_)           | |  
 | |__ ___  _ __ | |_ __ _| |_ _  ___  _ __ | |_ 
 |  __/ _ \| '_ \| __/ _\` | __| |/ _ \| '_ \| __|
 | | | (_) | |_) | || (_| | |_| | (_) | | | | |_ 
 |_|  \___/| .__/ \__\__,_|\__|_|\___/|_| |_|\__|
           | |                                    
           |_|                                    
${NC}"

# Create directory with visual feedback
echo -e "${YELLOW}[1/5] Preparing workspace...${NC}"
mkdir -pv "$WORK_DIR" | sed "s/^/    /"

# Download with animated progress
echo -e "${YELLOW}[2/5] Downloading dataset...${NC}"
curl -# -L -o "$ZIP_FILE" "$URL" 2>&1 | awk '{printf "\r    Progress: [%-50s] %d%%", substr($0,1,50), $2}'
echo -e "\n    Download complete!"

# Database cleanup with visual confirmation
echo -e "${YELLOW}[3/5] Checking for existing database...${NC}"
if [ -f "$DB_FILE" ]; then
    echo -e "    ${RED}Found existing database:${NC}"
    ls -lh "$DB_FILE" | awk '{print "    " $0}'
    echo -e "    ${RED}Removing...${NC}"
    rm -v "$DB_FILE" | sed "s/^/    /"
else
    echo -e "    ${GREEN}No existing database found${NC}"
fi

# Unzip with file counter
echo -e "${YELLOW}[4/5] Unzipping files...${NC}"
TOTAL_FILES=$(unzip -l "$ZIP_FILE" | tail -1 | awk '{print $2}')
echo -e "    ${BLUE}Found $TOTAL_FILES files in archive${NC}"
unzip -o "$ZIP_FILE" -d "$WORK_DIR" | awk -v total="$TOTAL_FILES" '
    BEGIN { count=0 }
    /inflating/ { count++; printf "\r    Extracting... [%-50s] %d%%", substr("##################################################",1,count/total*50), (count/total*100) }
    END { printf "\n" }'

# Run Python script with visual feedback
echo -e "${YELLOW}[5/5] Running data import...${NC}"
python3 "$WORK_DIR/import_all_data.py" 2>&1 | while read -r line; do
    echo -e "    ${BLUE}Python:${NC} $line"
done

# Completion message
echo -e "\n${GREEN}✓ All operations completed successfully!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
