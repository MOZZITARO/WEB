// ... existing code ...
function drawMatrix(matrix, offset, context) {
    matrix.forEach((row, y) => {
        row.forEach((value, x) => {
            if (value !== 0) {
                // 블록 그림자 효과
                context.fillStyle = 'rgba(0, 0, 0, 0.2)';
                context.fillRect(x + offset.x + 0.05, y + offset.y + 0.05, 0.9, 0.9);
                
                // 메인 블록 색상
                context.fillStyle = COLORS[value];
                context.fillRect(x + offset.x, y + offset.y, 0.9, 0.9);
                
                // 블록 테두리
                context.strokeStyle = '#000';
                context.lineWidth = 0.05;
                context.strokeRect(x + offset.x, y + offset.y, 0.9, 0.9);
                
                // 블록 하이라이트 효과
                context.fillStyle = 'rgba(255, 255, 255, 0.1)';
                context.fillRect(x + offset.x, y + offset.y, 0.9, 0.2);
                context.fillRect(x + offset.x, y + offset.y, 0.2, 0.9);
            }
        });
    });
}

function draw() {
    // 게임 보드 배경
    ctx.fillStyle = '#f0f0f0';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // 그리드 라인 그리기
    ctx.strokeStyle = '#ddd';
    ctx.lineWidth = 0.02;
    for (let i = 0; i <= BOARD_WIDTH; i++) {
        ctx.beginPath();
        ctx.moveTo(i, 0);
        ctx.lineTo(i, BOARD_HEIGHT);
        ctx.stroke();
    }
    for (let i = 0; i <= BOARD_HEIGHT; i++) {
        ctx.beginPath();
        ctx.moveTo(0, i);
        ctx.lineTo(BOARD_WIDTH, i);
        ctx.stroke();
    }
    
    drawMatrix(board, {x: 0, y: 0}, ctx);
    if (piece) {
        drawMatrix(piece.matrix, piece.pos, ctx);
    }
    
    // 다음 블록 미리보기 영역
    nextPieceCtx.fillStyle = '#f0f0f0';
    nextPieceCtx.fillRect(0, 0, nextPieceCanvas.width, nextPieceCanvas.height);
    if (nextPiece) {
        drawMatrix(nextPiece.matrix, {x: 1, y: 1}, nextPieceCtx);
    }
}
// ... existing code ...