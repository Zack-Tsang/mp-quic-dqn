package quic

import (
	"fmt"
	"encoding/csv"
	"os"
	"github.com/lucas-clemente/quic-go/internal/utils"
)

type OfflineWriter struct{
	buffer [][]string
}

func (o *OfflineWriter) Init(){
	o.buffer = [][]string{}
}

func (o *OfflineWriter) Append(row []string){
	o.buffer = append(o.buffer, row)
}

func (o *OfflineWriter) Close(finalReward string, id string){
	lastState := o.buffer[len(o.buffer)-1][1]
	o.Append([]string{finalReward, lastState, "END"})
	fileName := fmt.Sprintf("../data/episode_%s.csv", id)

	file, err := os.Create(fileName)
	if err != nil {
		utils.Errorf("error creating %s", fileName)
		panic(err)
	}
	writer := csv.NewWriter(file)
	defer file.Close()

	for _, row := range o.buffer{
		err := writer.Write(row)
		if err != nil{
			utils.Errorf("error writing row %s", row)
			panic(err)
		}
	}
	o.buffer = [][]string{}
}
